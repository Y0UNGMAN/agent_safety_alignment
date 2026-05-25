import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_tools(raw_tools):
    converted = []
    for tool in raw_tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}, "required": []}),
                },
            }
        )
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tools", default="data/tools/mock_tools.json")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    tools = convert_tools(read_json(args.tools))
    messages = [
        {
            "role": "system",
            "content": "你是一个具备工具调用能力的安全 Agent。只在任务需要且权限允许时调用工具。",
        },
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    print(tokenizer.decode(generated, skip_special_tokens=False))


if __name__ == "__main__":
    main()
