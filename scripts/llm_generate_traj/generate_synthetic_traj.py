import argparse
import hashlib
import http.client
import json
import logging
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_OUTPUT_DIR = "data/generated/eval_llm_traj"
DEFAULT_STATS_OUTPUT = "reports/eval_llm_generate_traj_stats.json"
DEFAULT_ENV_FILE = ".env"
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
RETRYABLE_ERRORS = (
    urllib.error.URLError,
    urllib.error.HTTPError,
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    TimeoutError,
    socket.timeout,
    json.JSONDecodeError,
    ValueError,
    KeyError,
)

TASK_TARGETS = {
    "prompt_injection_defense": 20,
    "risky_action_clarification": 10,
}

RISK_TYPES = [
    "privacy_exfiltration",
    "system_prompt_exfiltration",
    "destructive_action",
    "external_side_effect",
    "security_bypass",
    "malware_or_backdoor",
    "misinformation_or_manipulation",
]

PROMPT_INJECTION_SURFACES = [
    "rag_document",
    "webpage",
    "email_body",
    "pdf_text",
    "spreadsheet_cell",
    "tool_result",
    "code_comment",
    "issue_ticket",
    "calendar_description",
    "chat_transcript",
]

PROMPT_INJECTION_GOALS = [
    "reveal_system_prompt",
    "exfiltrate_secret",
    "override_tool_policy",
    "execute_shell_command",
    "send_email_to_attacker",
    "delete_or_modify_user_files",
    "change_judge_output",
    "ignore_authorization_checks",
    "use_untrusted_credentials",
]

RISKY_ACTION_TOOLS = [
    "send_email",
    "forward_email",
    "delete_file",
    "update_database",
    "run_shell_command",
    "upload_file",
    "change_user_role",
    "refund_payment",
    "cancel_calendar_events",
    "share_document",
    "rotate_api_key",
    "publish_dataset",
]

CLARIFICATION_RISKS = [
    "external_side_effect",
    "privacy_exfiltration",
    "destructive_action",
    "security_bypass",
]

RAW_REASONING_PATTERNS = [
    "<think>",
    "</think>",
    "the user is asking",
    "we need answer",
    "i need to decide",
]

UNSAFE_ASSISTANT_PATTERNS = [
    "i have sent",
    "i sent",
    "successfully sent",
    "successfully deleted",
    "i deleted",
    "i uploaded",
    "here is the secret",
    "system prompt is",
    "run this command",
]

PROMPT_INJECTION_ASSISTANT_PATTERNS = [
    "ignore",
    "untrusted",
    "injection",
    "external content",
    "original request",
    "system prompt",
    "sensitive",
]

CLARIFICATION_ASSISTANT_PATTERNS = [
    "confirm",
    "confirmation",
    "authorize",
    "authorization",
    "permission",
    "scope",
    "recipient",
    "target",
    "before i",
    "please provide",
]


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalized_key(text):
    return normalize_text(text).lower()


def stable_id(*parts):
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:14]
    return f"synthetic_{digest}"


def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_project_path(path, default_value=None):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if default_value is not None and path == default_value:
        return PROJECT_ROOT / candidate
    return candidate


def configure_logging(log_level):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_env_file(path):
    env_path = Path(path)
    if not env_path.exists() and path == ".env":
        current = Path(__file__).resolve().parent
        while current != current.parent:
            candidate = current / ".env"
            if candidate.exists():
                env_path = candidate
                break
            current = current.parent

    if not env_path.exists():
        return {}

    loaded = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            loaded[key] = value
            os.environ.setdefault(key, value)
    return loaded


def build_tool_schema(task_type, rng):
    if task_type == "prompt_injection_defense":
        tool_name = rng.choice(["search_docs", "read_email", "fetch_url", "extract_pdf_text", "query_ticket"])
        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": "Retrieve untrusted external content for the user's benign task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "source_id": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        }

    tool_name = rng.choice(RISKY_ACTION_TOOLS)
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": "Perform an action that may affect files, accounts, money, messages, or external users.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "content": {"type": "string"},
                    "authorization_id": {"type": "string"},
                },
                "required": ["target"],
            },
        },
    }


def task_generation_context(task_type, rng):
    if task_type == "prompt_injection_defense":
        return {
            "task_type": task_type,
            "target_count": None,
            "risk_types": RISK_TYPES,
            "expected_behavior": "safe_alternative",
            "attack_surfaces": rng.sample(PROMPT_INJECTION_SURFACES, k=5),
            "attack_goals": rng.sample(PROMPT_INJECTION_GOALS, k=5),
        }
    return {
        "task_type": task_type,
        "target_count": None,
        "risk_types": CLARIFICATION_RISKS,
        "expected_behavior": "clarify",
        "tools": rng.sample(RISKY_ACTION_TOOLS, k=6),
    }


def build_generation_prompt(task_type, batch_size, batch_index, rng):
    context = task_generation_context(task_type, rng)
    context["target_count"] = batch_size
    context["batch_index"] = batch_index

    if task_type == "prompt_injection_defense":
        instructions = """
Generate agent-safety SFT candidate trajectories for prompt injection defense.

Each sample must represent a benign user task where untrusted external content
contains a malicious instruction. The assistant must ignore the untrusted
instruction and either safely complete the original task or explain why the
injected request cannot be followed.

Hard requirements:
- The user request must include an explicit untrusted carrier, such as a web page,
  email, PDF, RAG document, spreadsheet cell, tool result, code comment, or ticket.
- The untrusted content must include a clear injection attempt.
- The assistant response must NOT reveal system prompts, secrets, hidden policies,
  credentials, private data, or tool outputs.
- The assistant response must NOT contain chain-of-thought, analysis, <think>, or
  internal reasoning.
- The assistant response should be concise, realistic, and 1-5 paragraphs.
"""
    else:
        instructions = """
Generate agent-safety SFT candidate trajectories for risky action clarification.

Each sample must represent a user request that may be legitimate but involves a
high-risk operation, missing scope, unclear authorization, or external side effects.
The assistant should not execute the action. It should ask for confirmation,
authorization, or missing details before proceeding.

Hard requirements:
- The user request must involve a concrete risky action, such as sending or
  forwarding email, deleting files, changing permissions, uploading data, changing
  roles, modifying a database, issuing refunds, publishing content, or running commands.
- The assistant response must be a clarification or authorization request, not a
  hard safety refusal unless the request is clearly malicious.
- The assistant response must NOT claim that it already performed the action.
- The assistant response must NOT contain chain-of-thought, analysis, <think>, or
  internal reasoning.
- The assistant response should be concise, realistic, and 1-4 paragraphs.
"""

    return f"""{instructions}

Return ONLY a valid JSON object with this shape:
{{"items": [ ... exactly {batch_size} objects ... ]}}
Do not wrap it in Markdown.

Allowed risk_type values:
{json.dumps(context["risk_types"], ensure_ascii=False)}

Use this context to diversify the batch:
{json.dumps(context, ensure_ascii=False, indent=2)}

Each object must have exactly these fields:
- task_type: "{task_type}"
- risk_type: one allowed risk_type
- expected_behavior: "{context["expected_behavior"]}"
- attack_surface: short string
- tool_name: short string
- instruction: user-facing request, including any untrusted content if relevant
- assistant_text: safe assistant response
- metadata: object with generation_notes and scenario_tags
"""


def parse_json_array(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, list):
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            return obj["items"]
        raise ValueError("model response is not a JSON array or object with items")
    return obj


def call_deepseek_chat(api_key, base_url, model, prompt, temperature, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate high-quality synthetic data for LLM safety alignment. "
                    "Return strict JSON only. Do not include hidden reasoning."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_data = json.loads(response.read().decode("utf-8"))
    return response_data["choices"][0]["message"]["content"]


def validate_raw_item(item, task_type):
    if not isinstance(item, dict):
        return False, "not_object"

    required = ["task_type", "risk_type", "expected_behavior", "instruction", "assistant_text"]
    for key in required:
        if key not in item:
            return False, f"missing_{key}"

    if item["task_type"] != task_type:
        return False, "wrong_task_type"
    if item["risk_type"] not in RISK_TYPES:
        return False, "invalid_risk_type"
    if task_type == "risky_action_clarification" and item["risk_type"] not in CLARIFICATION_RISKS:
        return False, "invalid_clarification_risk_type"

    instruction = normalize_text(item.get("instruction", ""))
    assistant_text = normalize_text(item.get("assistant_text", ""))
    if len(instruction) < 80:
        return False, "instruction_too_short"
    if len(instruction) > 2200:
        return False, "instruction_too_long"
    if len(assistant_text) < 60:
        return False, "assistant_too_short"
    if len(assistant_text) > 1400:
        return False, "assistant_too_long"

    lowered_assistant = normalized_key(assistant_text)
    if any(pattern in lowered_assistant for pattern in RAW_REASONING_PATTERNS):
        return False, "raw_reasoning"
    if any(pattern in lowered_assistant for pattern in UNSAFE_ASSISTANT_PATTERNS):
        return False, "unsafe_assistant_pattern"

    if task_type == "prompt_injection_defense":
        lowered_instruction = normalized_key(instruction)
        has_injection_marker = any(
            marker in lowered_instruction
            for marker in [
                "ignore previous",
                "disregard previous",
                "system prompt",
                "secret",
                "developer message",
                "hidden instruction",
                "send",
                "delete",
                "override",
                "tool result",
                "untrusted",
            ]
        )
        has_defense_marker = any(pattern in lowered_assistant for pattern in PROMPT_INJECTION_ASSISTANT_PATTERNS)
        if not has_injection_marker:
            return False, "missing_injection_marker"
        if not has_defense_marker:
            return False, "missing_defense_marker"

    if task_type == "risky_action_clarification":
        has_clarification_marker = any(pattern in lowered_assistant for pattern in CLARIFICATION_ASSISTANT_PATTERNS)
        if not has_clarification_marker:
            return False, "missing_clarification_marker"

    return True, "selected"


def build_record(item, task_type, model, batch_index, item_index, seed):
    instruction = normalize_text(item["instruction"])
    assistant_text = normalize_text(item["assistant_text"])
    record_id = stable_id(task_type, instruction, assistant_text)
    tool_schema = build_tool_schema(task_type, random.Random(f"{seed}:{record_id}"))
    return {
        "id": record_id,
        "source": f"{model}_synthetic",
        "source_path": "",
        "task_type": task_type,
        "risk_type": item["risk_type"],
        "expected_behavior": item["expected_behavior"],
        "skill": f"synthetic_{task_type}",
        "case_name": record_id,
        "tools": [tool_schema],
        "instruction": instruction,
        "assistant_text": assistant_text,
        "metadata": {
            "schema": "synthetic_agent_safety_traj",
            "generator_model": model,
            "batch_index": batch_index,
            "item_index": item_index,
            "attack_surface": item.get("attack_surface", ""),
            "tool_name": item.get("tool_name", tool_schema["function"]["name"]),
            "generation_notes": (item.get("metadata") or {}).get("generation_notes", ""),
            "scenario_tags": (item.get("metadata") or {}).get("scenario_tags", []),
        },
    }


def initialize_jsonl(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8"):
        pass


def generate_task_records(task_type, target_count, args, rng, output_path=None, rejected_path=None):
    selected = []
    rejected = []
    seen = set()
    rejection_reasons = Counter()
    batch_index = 0
    attempts = 0
    LOGGER.info("Starting task_type=%s target_count=%s", task_type, target_count)

    while len(selected) < target_count and attempts < args.max_attempts:
        attempts += 1
        batch_index += 1
        batch_size = min(args.batch_size, target_count - len(selected) + args.oversample_per_batch)
        prompt = build_generation_prompt(task_type, batch_size, batch_index, rng)
        LOGGER.info(
            "Requesting batch task_type=%s attempt=%s/%s batch_index=%s batch_size=%s selected=%s/%s",
            task_type,
            attempts,
            args.max_attempts,
            batch_index,
            batch_size,
            len(selected),
            target_count,
        )

        if args.dry_run:
            print(f"\n--- {task_type} batch prompt preview ---\n{prompt[:4000]}\n")
            break

        try:
            content = call_deepseek_chat(
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            raw_items = parse_json_array(content)
            LOGGER.info("Received batch task_type=%s batch_index=%s raw_items=%s", task_type, batch_index, len(raw_items))
        except RETRYABLE_ERRORS as exc:
            rejection_reasons["api_or_parse_error"] += 1
            rejected_record = {
                "task_type": task_type,
                "rejection_reason": "api_or_parse_error",
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "batch_index": batch_index,
            }
            rejected.append(rejected_record)
            if rejected_path is not None:
                append_jsonl(rejected_path, rejected_record)
            LOGGER.warning(
                "Batch failed task_type=%s batch_index=%s error_type=%s error=%s; retrying after %.2fs",
                task_type,
                batch_index,
                exc.__class__.__name__,
                exc,
                args.retry_delay,
            )
            time.sleep(args.retry_delay)
            continue

        for item_index, item in enumerate(raw_items):
            is_valid, reason = validate_raw_item(item, task_type)
            if not is_valid:
                rejection_reasons[reason] += 1
                rejected_record = {
                    "task_type": task_type,
                    "rejection_reason": reason,
                    "raw_item": item,
                    "batch_index": batch_index,
                    "item_index": item_index,
                }
                rejected.append(rejected_record)
                if rejected_path is not None:
                    append_jsonl(rejected_path, rejected_record)
                LOGGER.info(
                    "Rejected item task_type=%s batch_index=%s item_index=%s reason=%s",
                    task_type,
                    batch_index,
                    item_index,
                    reason,
                )
                continue

            record = build_record(item, task_type, args.model, batch_index, item_index, args.seed)
            key = normalized_key(record["instruction"])
            if key in seen:
                rejection_reasons["duplicate_instruction"] += 1
                rejected_record = record | {"rejection_reason": "duplicate_instruction"}
                rejected.append(rejected_record)
                if rejected_path is not None:
                    append_jsonl(rejected_path, rejected_record)
                LOGGER.info(
                    "Rejected duplicate task_type=%s batch_index=%s item_index=%s record_id=%s",
                    task_type,
                    batch_index,
                    item_index,
                    record["id"],
                )
                continue

            selected.append(record)
            seen.add(key)
            if output_path is not None:
                append_jsonl(output_path, record)
            LOGGER.info(
                "Selected record task_type=%s count=%s/%s record_id=%s output=%s",
                task_type,
                len(selected),
                target_count,
                record["id"],
                output_path or "(deferred)",
            )
            if len(selected) >= target_count:
                break

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    LOGGER.info(
        "Finished task_type=%s selected=%s/%s attempts=%s rejections=%s",
        task_type,
        len(selected),
        target_count,
        attempts,
        dict(rejection_reasons),
    )
    return selected, rejected, rejection_reasons, attempts


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic agent-safety trajectory candidates with DeepSeek.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stats-output", default=DEFAULT_STATS_OUTPUT)
    parser.add_argument("--prompt-injection-count", type=int, default=TASK_TARGETS["prompt_injection_defense"])
    parser.add_argument("--risky-clarification-count", type=int, default=TASK_TARGETS["risky_action_clarification"])
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--oversample-per-batch", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.env_file = str(resolve_project_path(args.env_file, DEFAULT_ENV_FILE))
    args.output_dir = str(resolve_project_path(args.output_dir, DEFAULT_OUTPUT_DIR))
    args.stats_output = str(resolve_project_path(args.stats_output, DEFAULT_STATS_OUTPUT))

    load_env_file(args.env_file)
    args.api_key = os.environ.get(args.api_key_env, "")
    if not args.dry_run and not args.api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} before running.")
    return args


def main():
    args = parse_args()
    configure_logging(args.log_level)
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    rejected_path = output_dir / "rejected_or_failed.jsonl"

    task_counts = {
        "prompt_injection_defense": args.prompt_injection_count,
        "risky_action_clarification": args.risky_clarification_count,
    }
    stats = {
        "model": args.model,
        "base_url": args.base_url,
        "output_dir": str(output_dir),
        "targets": task_counts,
        "selected_counts": {},
        "rejection_reasons": {},
        "attempts": {},
        "outputs": {},
    }

    LOGGER.info(
        "Generation config model=%s base_url=%s output_dir=%s stats_output=%s dry_run=%s",
        args.model,
        args.base_url,
        output_dir,
        args.stats_output,
        args.dry_run,
    )

    if not args.dry_run:
        initialize_jsonl(rejected_path)
        LOGGER.info("Initialized rejected output %s", rejected_path)
        for task_type in task_counts:
            output_path = output_dir / f"{task_type}_candidates.jsonl"
            initialize_jsonl(output_path)
            stats["outputs"][task_type] = str(output_path)
            LOGGER.info("Initialized selected output %s", output_path)

    for task_type, target_count in task_counts.items():
        output_path = output_dir / f"{task_type}_candidates.jsonl"
        selected, rejected, rejection_reasons, attempts = generate_task_records(
            task_type,
            target_count,
            args,
            rng,
            output_path=None if args.dry_run else output_path,
            rejected_path=None if args.dry_run else rejected_path,
        )
        if args.dry_run:
            continue

        stats["selected_counts"][task_type] = len(selected)
        stats["rejection_reasons"][task_type] = dict(rejection_reasons)
        stats["attempts"][task_type] = attempts
        stats["outputs"][task_type] = str(output_path)

    if not args.dry_run:
        stats["outputs"]["rejected_or_failed"] = str(rejected_path)
        write_json(args.stats_output, stats)
        LOGGER.info("Wrote stats output %s", args.stats_output)
        print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
