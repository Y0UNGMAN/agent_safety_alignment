import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFENSIVE_KEYWORDS = {
    "access control",
    "authentication",
    "authorization",
    "backup",
    "best practice",
    "compliance",
    "data protection",
    "defend",
    "defender",
    "defenders",
    "defense",
    "detect",
    "detection",
    "encryption",
    "hardening",
    "incident response",
    "input validation",
    "least privilege",
    "logging",
    "mitigate",
    "mitigation",
    "monitor",
    "patch",
    "pii",
    "policy",
    "prevent",
    "prevention",
    "privacy",
    "protect",
    "remediation",
    "risk management",
    "secure",
    "security audit",
    "security awareness",
    "security policy",
    "secret management",
    "token",
    "vulnerability management",
}


DANGEROUS_KEYWORDS = {
    "backdoor",
    "bruteforce",
    "brute force",
    "bypass authentication",
    "bypass detection",
    "crack password",
    "credential theft",
    "ddos",
    "dump credentials",
    "evade detection",
    "exfiltrate",
    "exploit",
    "keylogger",
    "malware",
    "payload",
    "phishing kit",
    "privilege escalation",
    "ransomware",
    "reverse shell",
    "rootkit",
    "sql injection payload",
    "steal",
    "trojan",
    "weaponize",
}


SAFE_CONTEXT_KEYWORDS = {
    "avoid",
    "defend",
    "detect",
    "detection",
    "hardening",
    "mitigate",
    "mitigation",
    "monitor",
    "patch",
    "prevent",
    "prevention",
    "protect",
    "remediate",
    "remediation",
    "secure",
}


REFUSAL_PATTERNS = (
    "i can't assist",
    "i cannot assist",
    "i can’t assist",
    "i cannot help",
    "i can't help",
    "i’m sorry",
    "i'm sorry",
    "sorry, but i",
)


DOMAIN_KEYWORDS = {
    "secure_development": {
        "code",
        "developer",
        "input validation",
        "parameterized",
        "query",
        "secure coding",
        "software",
        "sql injection",
        "testing",
        "vulnerability",
    },
    "privacy_and_data_protection": {
        "api key",
        "data protection",
        "personal data",
        "pii",
        "privacy",
        "secret",
        "sensitive data",
        "token",
    },
    "access_control_and_auth": {
        "access control",
        "authentication",
        "authorization",
        "least privilege",
        "mfa",
        "multi-factor",
        "password",
        "permission",
        "two-factor",
    },
    "security_operations": {
        "audit",
        "backup",
        "detect",
        "incident",
        "logging",
        "monitor",
        "patch",
        "remediation",
        "risk",
    },
    "agent_tool_safety_related": {
        "agent",
        "external tool",
        "prompt injection",
        "system prompt",
        "tool",
        "tool call",
        "untrusted",
    },
}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def contains_any(text, keywords):
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def is_dangerous_without_safe_context(user_text, assistant_text):
    combined = f"{user_text}\n{assistant_text}".lower()
    if not contains_any(combined, DANGEROUS_KEYWORDS):
        return False
    return not contains_any(combined, SAFE_CONTEXT_KEYWORDS)


def looks_like_refusal(assistant_text):
    lowered = assistant_text.lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


def too_code_heavy(text):
    code_markers = text.count("```")
    command_markers = len(re.findall(r"\b(curl|nc|ncat|netcat|msfconsole|powershell|chmod|rm -rf|wget)\b", text.lower()))
    return code_markers >= 2 or command_markers >= 3


def classify_domain(user_text, assistant_text):
    combined = f"{user_text}\n{assistant_text}".lower()
    scores = {
        domain: sum(1 for keyword in keywords if keyword in combined)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    return best_domain if best_score > 0 else "general_security"


def is_candidate(user_text, assistant_text, min_user_chars, max_user_chars, min_assistant_chars, max_assistant_chars):
    user_text = normalize_text(user_text)
    assistant_text = normalize_text(assistant_text)
    combined = f"{user_text}\n{assistant_text}"

    if not (min_user_chars <= len(user_text) <= max_user_chars):
        return False
    if not (min_assistant_chars <= len(assistant_text) <= max_assistant_chars):
        return False
    if not contains_any(combined, DEFENSIVE_KEYWORDS):
        return False
    if is_dangerous_without_safe_context(user_text, assistant_text):
        return False
    if looks_like_refusal(assistant_text):
        return False
    if too_code_heavy(combined):
        return False
    return True


def extract_cybersecurity_qa(row):
    instruction = normalize_text(row.get("instruction", ""))
    input_text = normalize_text(row.get("input", ""))
    output = normalize_text(row.get("output", ""))
    user_text = instruction if not input_text else f"{instruction}\n\n{input_text}"
    return user_text, output


def extract_cybersecdata(row):
    messages = row.get("messages", [])
    if not isinstance(messages, list):
        return "", ""

    user_text = ""
    assistant_text = ""
    for message in messages:
        role = message.get("role")
        content = normalize_text(message.get("content", ""))
        if role == "user" and not user_text:
            user_text = content
        elif role == "assistant":
            assistant_text = content
    return user_text, assistant_text


def select_candidates(path, source, extractor, target_size, args, seed_offset=0):
    seen_users = set()
    candidates = []
    rejected = Counter()

    for line_number, row in read_jsonl(path):
        user_text, assistant_text = extractor(row)
        user_key = user_text.lower()
        if not user_key:
            rejected["empty_user"] += 1
            continue
        if user_key in seen_users:
            rejected["duplicate_user"] += 1
            continue
        seen_users.add(user_key)

        if not is_candidate(
            user_text,
            assistant_text,
            args.min_user_chars,
            args.max_user_chars,
            args.min_assistant_chars,
            args.max_assistant_chars,
        ):
            rejected["quality_or_safety_filter"] += 1
            continue

        domain = classify_domain(user_text, assistant_text)
        candidates.append(
            {
                "line_number": line_number,
                "domain": domain,
                "record": row,
            }
        )

    rng = random.Random(args.seed + seed_offset)
    by_domain = defaultdict(list)
    for item in candidates:
        by_domain[item["domain"]].append(item)
    for rows in by_domain.values():
        rng.shuffle(rows)

    selected = []
    domain_order = sorted(by_domain.keys())
    while len(selected) < target_size and domain_order:
        progressed = False
        for domain in list(domain_order):
            if by_domain[domain]:
                selected.append(by_domain[domain].pop())
                progressed = True
                if len(selected) >= target_size:
                    break
            else:
                domain_order.remove(domain)
        if not progressed:
            break

    rng.shuffle(selected)
    selected_records = [item["record"] for item in selected]
    selected_meta = [
        {
            "source": source,
            "line_number": item["line_number"],
            "domain": item["domain"],
        }
        for item in selected
    ]

    stats = {
        "source": source,
        "input_path": path,
        "target_size": target_size,
        "candidate_count": len(candidates),
        "selected_count": len(selected_records),
        "selected_by_domain": dict(Counter(item["domain"] for item in selected)),
        "rejected": dict(rejected),
    }
    return selected_records, selected_meta, stats


def main():
    parser = argparse.ArgumentParser(description="Select defensive normal QA samples without converting their schema.")
    parser.add_argument("--cybersecurity-qa-path", default="original_dataset/cybersecurity_qa-bucket.jsonl")
    parser.add_argument("--cybersecdata-path", default="original_dataset/cybersecdata-bucket_train.jsonl")
    parser.add_argument("--output-dir", default="data/selected/normal_qa_safe_completion")
    parser.add_argument("--stats-path", default="reports/normal_qa_safe_completion_selection_stats.json")
    parser.add_argument("--cybersecurity-qa-size", type=int, default=120)
    parser.add_argument("--cybersecdata-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-user-chars", type=int, default=20)
    parser.add_argument("--max-user-chars", type=int, default=700)
    parser.add_argument("--min-assistant-chars", type=int, default=50)
    parser.add_argument("--max-assistant-chars", type=int, default=1800)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cyberqa_rows, cyberqa_meta, cyberqa_stats = select_candidates(
        args.cybersecurity_qa_path,
        "cybersecurity_qa",
        extract_cybersecurity_qa,
        args.cybersecurity_qa_size,
        args,
        seed_offset=0,
    )
    cybersec_rows, cybersec_meta, cybersec_stats = select_candidates(
        args.cybersecdata_path,
        "cybersecdata",
        extract_cybersecdata,
        args.cybersecdata_size,
        args,
        seed_offset=1000,
    )

    write_jsonl(output_dir / "cybersecurity_qa_selected.jsonl", cyberqa_rows)
    write_jsonl(output_dir / "cybersecdata_selected.jsonl", cybersec_rows)
    write_jsonl(output_dir / "selection_metadata.jsonl", cyberqa_meta + cybersec_meta)

    stats = {
        "total_selected": len(cyberqa_rows) + len(cybersec_rows),
        "outputs": {
            "cybersecurity_qa": str(output_dir / "cybersecurity_qa_selected.jsonl"),
            "cybersecdata": str(output_dir / "cybersecdata_selected.jsonl"),
            "metadata": str(output_dir / "selection_metadata.jsonl"),
        },
        "sources": {
            "cybersecurity_qa": cyberqa_stats,
            "cybersecdata": cybersec_stats,
        },
    }
    write_json(args.stats_path, stats)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
