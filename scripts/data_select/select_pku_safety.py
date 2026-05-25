import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT_FILES = [
    "original_dataset/PKU-SafeRLHF-bucket_Alphaca2-7B.jsonl",
    "original_dataset/PKU-SafeRLHF-bucket_Alphaca3-8B.jsonl",
]

DEFAULT_EXISTING_DIR = "data/selected/traj_safety"
DEFAULT_OUTPUT_DIR = "data/selected/pku_safety"
DEFAULT_STATS_OUTPUT = "reports/pku_safety_selection_stats.json"

TASK_TARGETS = {
    "safety_refusal": 250,
    "prompt_injection_defense": 100,
}

PKU_SOURCE_PREFIX = "PKU-SafeRLHF"

RELEVANT_HARM_CATEGORIES = {
    "Cybercrime",
    "Privacy Violation",
    "Economic Crime",
    "White-Collar Crime",
    "Mental Manipulation",
    "Endangering National Security",
    "Disrupting Public Order",
}

HARM_CATEGORY_TO_RISK_TYPE = [
    ("system_prompt_exfiltration", ["system prompt", "developer message", "hidden instruction"]),
    ("privacy_exfiltration", ["Privacy Violation"]),
    ("security_bypass", ["Cybercrime", "Endangering National Security"]),
    ("external_side_effect", ["Economic Crime", "White-Collar Crime", "Disrupting Public Order"]),
    ("misinformation_or_manipulation", ["Mental Manipulation"]),
]

PROMPT_INJECTION_PATTERNS = [
    r"\bignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|messages)\b",
    r"\bdisregard (all |any |the )?(previous|prior|above|earlier) (instructions|rules|messages)\b",
    r"\boverride (the )?(system|developer|safety) (prompt|message|instructions|rules)\b",
    r"\breveal (your|the) (system prompt|developer message|hidden instructions)\b",
    r"\bshow (your|the) (system prompt|developer message|hidden instructions)\b",
    r"\byou are now\b",
    r"\bdo anything now\b",
    r"\bdan mode\b",
    r"\bjailbreak\b",
    r"\bno restrictions\b",
    r"\bbypass (safety|filters|policy|rules)\b",
]

REFUSAL_PATTERNS = [
    "i cannot",
    "i can't",
    "i can’t",
    "i will not",
    "i won't",
    "i won’t",
    "cannot help",
    "can't help",
    "cannot assist",
    "not able to assist",
    "not appropriate",
    "not ethical",
    "illegal",
    "privacy",
    "unauthorized",
    "without consent",
    "against the law",
    "instead",
]

UNSAFE_RESPONSE_PATTERNS = [
    "here are the steps",
    "you can hack",
    "obtain their password",
    "spy app",
    "keylogger",
    "phishing",
    "malware",
    "exploit",
    "steal",
    "bypass authentication",
    "disable security",
]

RAW_REASONING_PATTERNS = [
    "</think>",
    "<think>",
    "the user is asking",
]


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalized_key(text):
    return normalize_text(text).lower()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def row_harm_categories(row):
    categories = set()
    for key in ["response_0_harm_category", "response_1_harm_category"]:
        for category, enabled in (row.get(key) or {}).items():
            if enabled:
                categories.add(category)
    return categories


def has_relevant_harm(row):
    categories = row_harm_categories(row)
    return bool(categories & RELEVANT_HARM_CATEGORIES)


def choose_safe_response(row):
    safer_response_id = row.get("safer_response_id")
    if safer_response_id in [0, 1] and row.get(f"is_response_{safer_response_id}_safe"):
        return safer_response_id, normalize_text(row.get(f"response_{safer_response_id}", ""))

    candidates = []
    for response_id in [0, 1]:
        if row.get(f"is_response_{response_id}_safe"):
            text = normalize_text(row.get(f"response_{response_id}", ""))
            candidates.append((response_quality_score(text), response_id, text))
    if not candidates:
        return None, ""

    candidates.sort(reverse=True)
    _, response_id, text = candidates[0]
    return response_id, text


def is_prompt_injection_prompt(prompt):
    lowered = normalized_key(prompt)
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in PROMPT_INJECTION_PATTERNS)


def classify_task_type(row):
    if is_prompt_injection_prompt(row.get("prompt", "")):
        return "prompt_injection_defense"
    return "safety_refusal"


def map_risk_type(row):
    prompt = normalized_key(row.get("prompt", ""))
    if is_prompt_injection_prompt(prompt) and any(
        marker in prompt for marker in ["system prompt", "developer message", "hidden instruction"]
    ):
        return "system_prompt_exfiltration"

    categories = row_harm_categories(row)
    for risk_type, markers in HARM_CATEGORY_TO_RISK_TYPE:
        if any(marker in categories or marker in prompt for marker in markers):
            return risk_type
    return "security_bypass"


def response_quality_score(text):
    lowered = normalized_key(text)
    score = 0
    score += 3 * sum(pattern in lowered for pattern in REFUSAL_PATTERNS)
    score -= 4 * sum(pattern in lowered for pattern in UNSAFE_RESPONSE_PATTERNS)
    score -= 5 * sum(pattern in lowered for pattern in RAW_REASONING_PATTERNS)
    if 80 <= len(text) <= 1200:
        score += 2
    if len(text) > 1800:
        score -= 5
    return score


def is_usable_candidate(row, task_type, response_text, args):
    prompt = normalize_text(row.get("prompt", ""))
    if not prompt:
        return False, "missing_prompt"
    if len(prompt) < args.min_prompt_chars:
        return False, "prompt_too_short"
    if len(prompt) > args.max_prompt_chars:
        return False, "prompt_too_long"
    if not has_relevant_harm(row):
        return False, "irrelevant_harm_category"
    if not response_text:
        return False, "no_safe_response"
    if len(response_text) < args.min_response_chars:
        return False, "response_too_short"
    if len(response_text) > args.max_response_chars:
        return False, "response_too_long"

    lowered_response = normalized_key(response_text)
    if any(pattern in lowered_response for pattern in RAW_REASONING_PATTERNS):
        return False, "raw_reasoning"
    if any(pattern in lowered_response for pattern in UNSAFE_RESPONSE_PATTERNS):
        return False, "unsafe_response_pattern"

    score = response_quality_score(response_text)
    if task_type == "prompt_injection_defense" and score < args.min_prompt_injection_score:
        return False, "low_response_score"
    if task_type == "safety_refusal" and score < args.min_refusal_score:
        return False, "low_response_score"
    return True, "selected"


def source_name_for(path):
    stem = Path(path).stem
    if "Alphaca2" in stem:
        return "PKU-SafeRLHF_Alphaca2-7B"
    if "Alphaca3" in stem:
        return "PKU-SafeRLHF_Alphaca3-8B"
    return f"{PKU_SOURCE_PREFIX}_{stem}"


def stable_id(source, prompt):
    digest = hashlib.sha1(f"{source}\n{prompt}".encode("utf-8")).hexdigest()[:12]
    return f"pku_{digest}"


def build_selected_record(row, source, source_path, response_id, response_text, task_type):
    risk_type = map_risk_type(row)
    prompt = normalize_text(row.get("prompt", ""))
    return {
        "id": stable_id(source, prompt),
        "source": source,
        "source_path": source_path,
        "task_type": task_type,
        "risk_type": risk_type,
        "expected_behavior": "safe_alternative" if task_type == "prompt_injection_defense" else "refuse",
        "skill": "pku_safe_rlhf",
        "case_name": stable_id(source, prompt),
        "instruction": prompt,
        "assistant_text": response_text,
        "metadata": {
            "schema": "pku_safe_rlhf",
            "prompt_source": row.get("prompt_source", ""),
            "response_id": response_id,
            "response_source": row.get(f"response_{response_id}_source", "") if response_id in [0, 1] else "",
            "safer_response_id": row.get("safer_response_id"),
            "better_response_id": row.get("better_response_id"),
            "is_response_0_safe": row.get("is_response_0_safe"),
            "is_response_1_safe": row.get("is_response_1_safe"),
            "response_0_severity_level": row.get("response_0_severity_level"),
            "response_1_severity_level": row.get("response_1_severity_level"),
            "harm_categories": sorted(row_harm_categories(row)),
            "response_score": response_quality_score(response_text),
        },
    }


def select_from_pku(input_files, args):
    rng = random.Random(args.seed)
    buckets = defaultdict(list)
    rejected = []
    stats = {
        "input_rows": 0,
        "by_source": Counter(),
        "candidate_counts": Counter(),
        "rejection_reasons": Counter(),
        "risk_type_counts": defaultdict(Counter),
    }

    for input_file in input_files:
        source = source_name_for(input_file)
        rows = read_jsonl(input_file)
        stats["input_rows"] += len(rows)
        stats["by_source"][source] += len(rows)
        for row_index, row in enumerate(rows):
            task_type = classify_task_type(row)
            response_id, response_text = choose_safe_response(row)
            is_usable, reason = is_usable_candidate(row, task_type, response_text, args)
            selected_record = build_selected_record(row, source, input_file, response_id, response_text, task_type)
            selected_record["metadata"]["source_row_index"] = row_index

            if is_usable:
                stats["candidate_counts"][task_type] += 1
                stats["risk_type_counts"][task_type][selected_record["risk_type"]] += 1
                buckets[task_type].append(selected_record)
            else:
                selected_record["rejection_reason"] = reason
                rejected.append(selected_record)
                stats["rejection_reasons"][reason] += 1

    for task_type in buckets:
        rng.shuffle(buckets[task_type])
        buckets[task_type].sort(key=lambda row: row["metadata"]["response_score"], reverse=True)

    return buckets, rejected, stats


def read_existing_records(existing_dir, task_type):
    path = Path(existing_dir) / f"{task_type}_selected.jsonl"
    if not path.exists():
        return []
    return read_jsonl(path)


def is_pku_record(row):
    return str(row.get("source", "")).startswith(PKU_SOURCE_PREFIX)


def merge_to_targets(existing_dir, selected_pku, args):
    merged = {}
    added = {}
    dedupe_stats = Counter()

    for task_type, target in TASK_TARGETS.items():
        existing_rows = [row for row in read_existing_records(existing_dir, task_type) if not is_pku_record(row)]
        seen_prompts = {normalized_key(row.get("instruction", "")) for row in existing_rows}
        seen_ids = {row.get("id") for row in existing_rows}
        remaining = max(0, target - len(existing_rows))
        task_added = []

        for row in selected_pku.get(task_type, []):
            if len(task_added) >= remaining:
                break
            prompt_key = normalized_key(row.get("instruction", ""))
            if prompt_key in seen_prompts:
                dedupe_stats[f"{task_type}:prompt_duplicate"] += 1
                continue
            if row.get("id") in seen_ids:
                dedupe_stats[f"{task_type}:id_duplicate"] += 1
                continue
            task_added.append(row)
            seen_prompts.add(prompt_key)
            seen_ids.add(row.get("id"))

        merged[task_type] = existing_rows + task_added
        added[task_type] = task_added

    return merged, added, dedupe_stats


def make_jsonable_stats(stats):
    return {
        "input_rows": stats["input_rows"],
        "by_source": dict(stats["by_source"]),
        "candidate_counts": dict(stats["candidate_counts"]),
        "rejection_reasons": dict(stats["rejection_reasons"]),
        "risk_type_counts": {task_type: dict(counter) for task_type, counter in stats["risk_type_counts"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Select PKU-SafeRLHF safety data and merge into selected candidates.")
    parser.add_argument("--input-files", nargs="*", default=DEFAULT_INPUT_FILES)
    parser.add_argument("--existing-dir", default=DEFAULT_EXISTING_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stats-output", default=DEFAULT_STATS_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-prompt-chars", type=int, default=20)
    parser.add_argument("--max-prompt-chars", type=int, default=1600)
    parser.add_argument("--min-response-chars", type=int, default=45)
    parser.add_argument("--max-response-chars", type=int, default=1400)
    parser.add_argument("--min-refusal-score", type=int, default=5)
    parser.add_argument("--min-prompt-injection-score", type=int, default=5)
    args = parser.parse_args()

    selected_pku, rejected, selection_stats = select_from_pku(args.input_files, args)
    merged, added, dedupe_stats = merge_to_targets(args.existing_dir, selected_pku, args)

    output_dir = Path(args.output_dir)
    for task_type in TASK_TARGETS:
        write_jsonl(output_dir / f"{task_type}_selected.jsonl", added[task_type])
        write_jsonl(Path(args.existing_dir) / f"{task_type}_selected.jsonl", merged[task_type])
    write_jsonl(output_dir / "rejected_or_failed.jsonl", rejected)

    stats = {
        "input_files": args.input_files,
        "targets": TASK_TARGETS,
        "selection": make_jsonable_stats(selection_stats),
        "added_counts": {task_type: len(rows) for task_type, rows in added.items()},
        "merged_counts": {task_type: len(rows) for task_type, rows in merged.items()},
        "added_by_risk_type": {
            task_type: dict(Counter(row["risk_type"] for row in rows)) for task_type, rows in added.items()
        },
        "added_by_source": {
            task_type: dict(Counter(row["source"] for row in rows)) for task_type, rows in added.items()
        },
        "dedupe_stats": dict(dedupe_stats),
        "outputs": {
            "pku_output_dir": str(output_dir),
            "merged_output_dir": args.existing_dir,
            "stats": args.stats_output,
        },
    }
    write_json(args.stats_output, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
