import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


# Edit these values directly when you want to report a different training run.
DEFAULT_OUTPUT_DIR = "outputs/sft_model_qwen3_8b"
DEFAULT_REPORT_ROOT = "reports/training_runs"
CREATE_TIMESTAMPED_REPORT_DIR = True


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def find_trainer_state(output_dir):
    output_path = Path(output_dir)
    root_state = output_path / "trainer_state.json"
    if root_state.exists():
        return root_state

    candidates = sorted(
        output_path.glob("checkpoint-*/trainer_state.json"),
        key=lambda path: int(path.parent.name.split("-")[-1]) if path.parent.name.split("-")[-1].isdigit() else -1,
    )
    if not candidates:
        raise FileNotFoundError(f"No trainer_state.json found under {output_dir}")
    return candidates[-1]


def extract_logs(state):
    return [row for row in state.get("log_history", []) if "loss" in row]


def metric_summary(logs, key):
    values = [float(row[key]) for row in logs if key in row]
    if not values:
        return None
    return {
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
    }


def format_number(value):
    if value is None:
        return ""
    if abs(value) < 0.001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.6g}"


def write_metrics_csv(path, logs):
    fields = ["step", "epoch", "loss", "grad_norm", "learning_rate", "mean_token_accuracy", "entropy", "num_tokens"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in logs:
            writer.writerow({key: row.get(key, "") for key in fields})


def build_markdown_summary(model_name, trainer_state_path, state, logs, summaries):
    lines = [
        f"# {model_name} Training Summary",
        "",
        f"- Source state: `{trainer_state_path}`",
        f"- Global step: {state.get('global_step')}",
        f"- Epoch: {state.get('epoch')}",
        f"- Logged points: {len(logs)}",
    ]
    if logs and logs[-1].get("num_tokens") is not None:
        lines.append(f"- Final logged tokens: {logs[-1].get('num_tokens')}")

    train_summaries = [row for row in state.get("log_history", []) if "train_runtime" in row]
    if train_summaries:
        train_summary = train_summaries[-1]
        lines.extend(
            [
                f"- Train runtime seconds: {train_summary.get('train_runtime')}",
                f"- Train loss: {train_summary.get('train_loss')}",
                f"- Train samples/sec: {train_summary.get('train_samples_per_second')}",
                f"- Train steps/sec: {train_summary.get('train_steps_per_second')}",
            ]
        )

    lines.extend(
        [
            "",
            "| metric | first | last | min | max | avg |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric, summary in summaries.items():
        if summary is None:
            continue
        lines.append(
            "| {metric} | {first} | {last} | {minv} | {maxv} | {avg} |".format(
                metric=metric,
                first=format_number(summary["first"]),
                last=format_number(summary["last"]),
                minv=format_number(summary["min"]),
                maxv=format_number(summary["max"]),
                avg=format_number(summary["avg"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Training-fit metrics show whether SFT optimization ran normally.",
            "- Loss/accuracy curves do not prove safety alignment quality by themselves.",
            "- Final claims should be based on held-out safety and utility evaluation.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_svg_curve(model_name, logs, csv_path):
    series_defs = [
        ("loss", "#2563eb"),
        ("mean_token_accuracy", "#16a34a"),
        ("grad_norm", "#dc2626"),
        ("learning_rate", "#9333ea"),
    ]
    width, height = 1100, 760
    margin_l, margin_r, margin_t, margin_b = 70, 30, 55, 55
    plot_w = width - margin_l - margin_r
    plot_h = 145
    row_gap = 35

    steps = [float(row["step"]) for row in logs]
    min_step, max_step = min(steps), max(steps)

    def scale_x(step):
        if max_step == min_step:
            return margin_l + plot_w / 2
        return margin_l + (step - min_step) / (max_step - min_step) * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            "<style>"
            "text{font-family:Arial,Helvetica,sans-serif;font-size:13px;fill:#111827}"
            ".title{font-size:22px;font-weight:700}.subtitle{font-size:13px;fill:#4b5563}"
            ".axis{stroke:#d1d5db;stroke-width:1}.grid{stroke:#eef2f7;stroke-width:1}"
            ".label{font-weight:700}.tick{font-size:11px;fill:#6b7280}"
            "</style>"
        ),
        f'<text x="70" y="32" class="title">{model_name} Training Curves</text>',
        '<text x="70" y="52" class="subtitle">Training metrics extracted from trainer_state.json</text>',
    ]

    for idx, (key, color) in enumerate(series_defs):
        if key not in logs[0]:
            continue
        values = [float(row[key]) for row in logs if key in row]
        y0 = margin_t + idx * (plot_h + row_gap)
        min_value, max_value = min(values), max(values)
        pad = (max_value - min_value) * 0.08 if max_value != min_value else 1.0
        low, high = min_value - pad, max_value + pad

        def scale_y(value):
            return y0 + plot_h - (value - low) / (high - low) * plot_h

        for grid_idx in range(5):
            y = y0 + grid_idx * plot_h / 4
            parts.append(f'<line x1="{margin_l}" y1="{y:.2f}" x2="{margin_l + plot_w}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<line x1="{margin_l}" y1="{y0 + plot_h:.2f}" x2="{margin_l + plot_w}" y2="{y0 + plot_h:.2f}" class="axis"/>')
        parts.append(f'<line x1="{margin_l}" y1="{y0:.2f}" x2="{margin_l}" y2="{y0 + plot_h:.2f}" class="axis"/>')

        points = " ".join(f"{scale_x(step):.2f},{scale_y(value):.2f}" for step, value in zip(steps, values))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{points}"/>')
        parts.append(f'<text x="{margin_l}" y="{y0 - 10}" class="label" fill="{color}">{key}</text>')
        parts.append(
            f'<text x="{margin_l + plot_w - 300}" y="{y0 - 10}" class="subtitle">'
            f'first {format_number(values[0])} | last {format_number(values[-1])} | '
            f'min {format_number(min_value)} | max {format_number(max_value)}</text>'
        )
        parts.append(f'<text x="18" y="{y0 + 14}" class="tick">{format_number(high)}</text>')
        parts.append(f'<text x="18" y="{y0 + plot_h:.2f}" class="tick">{format_number(low)}</text>')

    parts.append(f'<text x="{margin_l}" y="{height - 18}" class="subtitle">step {int(min_step)} to {int(max_step)}; CSV: {csv_path}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def make_report_dir(report_root, model_name, timestamp=None):
    if not CREATE_TIMESTAMPED_REPORT_DIR:
        return Path(report_root) / model_name
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(report_root) / f"{timestamp}_{model_name}"


def generate_report(output_dir, report_dir=None, run_name=None):
    output_path = Path(output_dir)
    model_name = run_name or output_path.name
    report_path = Path(report_dir) if report_dir else make_report_dir(DEFAULT_REPORT_ROOT, model_name)
    figures_path = report_path / "figures"
    trainer_state_path = find_trainer_state(output_path)
    state = read_json(trainer_state_path)
    logs = extract_logs(state)
    if not logs:
        raise ValueError(f"No logged loss entries found in {trainer_state_path}")

    summaries = {
        metric: metric_summary(logs, metric)
        for metric in ["loss", "grad_norm", "learning_rate", "mean_token_accuracy", "entropy"]
    }

    csv_path = report_path / f"training_metrics_{model_name}.csv"
    md_path = report_path / f"training_summary_{model_name}.md"
    svg_path = figures_path / f"{model_name}_training_curves.svg"

    write_metrics_csv(csv_path, logs)
    write_text(md_path, build_markdown_summary(model_name, trainer_state_path, state, logs, summaries))
    write_text(svg_path, build_svg_curve(model_name, logs, csv_path))

    return {
        "trainer_state": str(trainer_state_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "svg": str(svg_path),
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "logged_points": len(logs),
        "summaries": summaries,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate training reports from Hugging Face Trainer output.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    result = generate_report(args.output_dir, args.report_dir, args.run_name)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
