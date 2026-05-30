#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

TARGET_MODEL="${TARGET_MODEL:-qwen3-8b-base}"
TARGET_BASE_URL="${TARGET_BASE_URL:-http://localhost:8000/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.deepseek.com/v1}"
ENV_FILE="${ENV_FILE:-.env}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
SKIP_ENDPOINT_CHECK="${SKIP_ENDPOINT_CHECK:-0}"
SEND_TOOLS_TO_TARGET_API="${SEND_TOOLS_TO_TARGET_API:-0}"

echo "Project root: ${PROJECT_ROOT}"
echo "Target model: ${TARGET_MODEL}"
echo "Target base URL: ${TARGET_BASE_URL}"
echo "Judge model: ${JUDGE_MODEL}"
echo "Send tools to target API: ${SEND_TOOLS_TO_TARGET_API}"
echo

if [[ "${SKIP_ENDPOINT_CHECK}" != "1" ]]; then
  echo "[1/2] Checking target model endpoint..."
  export TARGET_MODEL
  export TARGET_BASE_URL
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

target_model = os.environ.get("TARGET_MODEL", "agent-safety-lora")
base_url = os.environ.get("TARGET_BASE_URL", "http://localhost:8000/v1").rstrip("/")
url = base_url + "/models"
try:
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
    print(f"Target endpoint check failed: {exc}", file=sys.stderr)
    print("Start vLLM first, or set SKIP_ENDPOINT_CHECK=1 if you know the endpoint is reachable.", file=sys.stderr)
    sys.exit(1)

models = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
print("Available models:", ", ".join(models) if models else "(no model ids returned)")
if target_model not in models:
    print(f"Target model '{target_model}' is not served by vLLM.", file=sys.stderr)
    print("Fix by deploying the LoRA model, or run with TARGET_MODEL set to one of the available model ids.", file=sys.stderr)
    sys.exit(1)
PY
else
  echo "[1/2] Skipping endpoint check."
fi
echo

echo "[2/2] Running offline eval with LLM judge..."
CMD=(
  "${PYTHON_BIN}" scripts/eval_offline/run_llm_judge_eval.py
  --target-model "${TARGET_MODEL}"
  --target-base-url "${TARGET_BASE_URL}"
  --judge-model "${JUDGE_MODEL}"
  --judge-base-url "${JUDGE_BASE_URL}"
  --env-file "${ENV_FILE}"
)

if [[ "${EVAL_LIMIT}" != "0" ]]; then
  CMD+=(--limit "${EVAL_LIMIT}")
fi

if [[ "${SEND_TOOLS_TO_TARGET_API}" == "1" ]]; then
  CMD+=(--send-tools-to-target-api)
fi

"${CMD[@]}"

echo
echo "Done. Check reports/eval_runs/ for results.jsonl, failures.jsonl, summary.json, and config.json."
