# --- 如何加载和使用训练好的 LoRA 模型 (示例) ---
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "DeepSeek-R1-Distill-Qwen-7B"
adapter_path = "sft_lora_model_int8_output" # 你保存 adapter 的路径

# 1. 加载基础模型 (可以再次量化加载，也可以不量化)
bnb_config_inf = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config_inf,
    device_map="auto",
    trust_remote_code=True
)
tokenizer_inf = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True) # 从 adapter 目录加载 tokenizer
if tokenizer_inf.pad_token is None:
    tokenizer_inf.pad_token = tokenizer_inf.eos_token

# 2. 加载 LoRA adapter 并合并到基础模型
model_inf = PeftModel.from_pretrained(base_model, adapter_path)
# model_inf = model_inf.merge_and_unload() # 可选：如果你想完全合并权重（需要足够内存，且会失去量化优势，除非基础模型已量化）

# 3. 使用模型进行推理
prompt = "Human: 一个病人反复咳嗽，咳痰带血丝，伴有午后低热、盗汗，最可能的诊断是什么？\nAssistant (Complex Reasoning): 这个病人有典型的肺结核症状：咳嗽、咳血痰、低热、盗汗。这些是肺结核的经典表现。\nAssistant (Final Response):"
inputs = tokenizer_inf(prompt, return_tensors="pt").to(model_inf.device)
outputs = model_inf.generate(**inputs, max_new_tokens=100, eos_token_id=tokenizer_inf.eos_token_id)
response = tokenizer_inf.decode(outputs[0], skip_special_tokens=True)
print("\n--- Inference Example ---")
print(response)