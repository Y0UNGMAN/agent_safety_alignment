import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_prepare.prepare_xlam_local import convert_row, is_usable


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_excluded_ids(paths):
    excluded = set()
    for path in paths:
        if not path or not Path(path).exists():
            continue
        for row in read_jsonl(path):
            if row.get("id"):
                excluded.add(row["id"])
    return excluded


def expected_tool_calls_from_messages(messages):
    for message in messages:
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            continue
        expected = []
        for call in tool_calls:
            function = call.get("function", {})
            arguments = function.get("arguments", "{}")
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                arguments = {}
            expected.append({"name": function.get("name"), "arguments": arguments})
        return expected
    return []


def to_eval_row(row):
    expected_tool_calls = expected_tool_calls_from_messages(row["messages"])
    return {
        **row,
        "expected_tool_calls": expected_tool_calls,
        "judge_type": "tool_call_exact_match",
        "metadata": {
            **row.get("metadata", {}),
            "eval_source": "xlam_holdout",
            "eval_notes": "Offline function-calling evaluation; tools are not executed.",
        },
    }


def balanced_sample(rows, total_size, seed):
    rng = random.Random(seed)
    buckets = {
        1: [row for row in rows if row["metadata"]["num_tool_calls"] == 1],
        2: [row for row in rows if row["metadata"]["num_tool_calls"] == 2],
        3: [row for row in rows if row["metadata"]["num_tool_calls"] == 3],
    }
    for bucket in buckets.values():
        rng.shuffle(bucket)

    targets = {
        1: int(total_size * 0.625),
        2: int(total_size * 0.25),
        3: total_size - int(total_size * 0.625) - int(total_size * 0.25),
    }
    selected = []
    for call_count, target in targets.items():
        selected.extend(buckets[call_count][:target])

    if len(selected) < total_size:
        selected_ids = {row["id"] for row in selected}
        leftovers = [row for row in rows if row["id"] not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: total_size - len(selected)])

    rng.shuffle(selected)
    return selected[:total_size]


def main():
    parser = argparse.ArgumentParser(description="Select xLAM holdout normal-tool-use eval data.")
    parser.add_argument("--input", default="original_dataset/xlam_function_calling_60k.json")
    parser.add_argument("--exclude", nargs="*", default=["data/processed/xlam_normal_tool_use.jsonl"])
    parser.add_argument("--output", default="data/eval/normal_tool_use_eval.jsonl")
    parser.add_argument("--stats-output", default="reports/eval_normal_tool_use_selection_stats.json")
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-tool-calls", type=int, default=1)
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--max-prompt-chars", type=int, default=12000)
    args = parser.parse_args()

    excluded_ids = load_excluded_ids(args.exclude)
    raw_rows = read_json(args.input)
    converted = []
    skipped = Counter()

    for raw_row in raw_rows:
        candidate_id = f"xlam_{raw_row.get('id')}"
        if candidate_id in excluded_ids:
            skipped["excluded_train_id"] += 1
            continue
        if not is_usable(raw_row, args.min_tool_calls, args.max_tool_calls, args.max_prompt_chars):
            skipped["not_usable"] += 1
            continue
        converted.append(to_eval_row(convert_row(raw_row)))

    selected = balanced_sample(converted, args.sample_size, args.seed)
    write_jsonl(args.output, selected)

    stats = {
        "input": args.input,
        "output": args.output,
        "sample_size": args.sample_size,
        "loaded_rows": len(raw_rows),
        "excluded_ids": len(excluded_ids),
        "usable_holdout_rows": len(converted),
        "written_rows": len(selected),
        "tool_call_distribution": dict(Counter(row["metadata"]["num_tool_calls"] for row in selected)),
        "skipped": dict(skipped),
    }
    write_json(args.stats_output, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
