import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(command):
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data and train the LoRA adapter.")
    parser.add_argument("--data-config", default="configs/data/sft_custom.yaml")
    parser.add_argument("--train-config", default="configs/train/sft_lora.yaml")
    parser.add_argument("--model-config", default="configs/model/qwen2_5_1_5b_lora.yaml")
    args = parser.parse_args()

    run_command(
        [
            sys.executable,
            "scripts/prepare_data.py",
            "--config",
            args.data_config,
        ]
    )
    run_command(
        [
            sys.executable,
            "scripts/train_sft.py",
            "--train-config",
            args.train_config,
            "--model-config",
            args.model_config,
        ]
    )


if __name__ == "__main__":
    main()
