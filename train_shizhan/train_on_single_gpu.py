import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


# --------------- 1.配置模型和数据路径 ---------------
model_id_or_path = "DeepSeek-R1-Distill-Qwen-7B"
dataset_path="dataset/medical_o1_sft_Chinese.json"
output_dir="sft_lora_model_int8_output_new"

tokenizer=AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
# model=AutoModelForCausalLM.from_pretrained(model_id_or_path,trust_remote_code=True)

# ----------------- 2.加载数据集 ---------------
print("Loading Dataset...")
#如果是多个数据集呢
dataset=load_dataset("json",data_files=dataset_path,split="train")
"""
Dataset 对象:
索引 (Row),Question (列1),Complex_CoT (列2),Response (列3)
0,""患者突发胸痛...""","""首先考虑心梗，需要...""","""建议立即进行心电图..."""
1,"""小儿发热39度...""","""患儿年龄小，高热需排除...""","""物理降温，口服退热药..."""
...,...,...,...
# 取第 0 行数据
#print(dataset[0]) 
# 输出结果是一个普通的 Python 字典：
# {
#   "Question": "患者突发胸痛...", 
#   "Complex_CoT": "首先考虑心梗...", 
#   "Response": "建议立即..."
# }

# 取出名为 "Question" 的整列数据
# print(dataset["Question"]) 

# 输出结果是一个包含所有问题的 Python 列表：
# ["患者突发胸痛...", "小儿发热39度...", "肚子疼怎么办...", ...]

"""
"""
print(f"数据集一共有{len(dataset)}组数据")
print('-'*100)
print(f"数据集列：",dataset.column_names)
print('-'*100)
print(f"数据集：",dataset)


# ------------------ 3.处理数据集 ---------------
# 这一步的作用是把输入输出拼接起来作为一个text输入给模型训练，并且在末尾要加一个eos-token，当然很多时候trl库会自动加，但是还是加了之后更为保险
train_prompt_style_zh = """
请根据下方指令和提供的具体问题，撰写一个恰当的回复。
在回答前，请仔细思考问题，并展现一步一步的思考过程，以确保回复的逻辑性和准确性。

### 指令：
你是一位医学专家，精通临床推理、诊断和治疗计划。
请回答下面的医学问题。

### 问题：
{}

### 回答：
<think>
{}
</think>
{}
"""
EOS_TOKEN = tokenizer.eos_token  # 一定要添加结束标记，否则会无限生成
def formatting_prompts_func(examples):
    inputs=examples['Question']
    cots=examples["Complex_CoT"]
    outputs=examples["Response"]
    texts=[]
    for input,cot,output in zip(inputs,cots,outputs):
        text=train_prompt_style_zh.format(input,cot,output)+EOS_TOKEN
        texts.append(text)
    return {
        "text":texts,
    }
    
# 使用 .map 应用格式化函数，创建 'text' 列,batched=True很关键
print("Formatting dataset...")
dataset = dataset.map(formatting_prompts_func,batched=True)
print("Dataset after formatting:", dataset)
# dataset 中增加一个text列
print("First example text:", dataset[0]['text'])
        
    

# 重要：设置 Padding Token
# SFTTrainer 和模型需要一个 Padding Token
# 对于很多 Causal LM，通常将 pad_token 设置为 eos_token
if tokenizer.pad_token is None:
    print("Tokenizer does not have a pad token, setting it to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token
    # 更新模型的 config，虽然 SFTTrainer 可能也会处理
    # model.config.pad_token_id = tokenizer.pad_token_id


################################################
# --------------- 4.模型量化 --------------------
################################################
print("Configuring BitsAndBytes for int8 quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # 可选: 对于更激进的量化（如 4bit），可以设置：
    # load_in_4bit=True,
    # bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16 # 根据 GPU 支持选择 bf16 或 fp16
)


print("Loading base model with int8 quantization...")
model=AutoModelForCausalLM.from_pretrained(
    model_id_or_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True 
)
print("Model Loaded.")
print(model)


################################################
# --------------- 5.lora配置 --------------------
################################################
print("Configuring LoRA...")

model=prepare_model_for_kbit_training(model)
lora_config=LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM"
)
# 使用 get_peft_model 应用 LoRA 配置（或者直接传给 SFTTrainer）
# SFTTrainer 会自动处理，所以这一步可以省略，直接将 lora_config 传给 Trainer
model = get_peft_model(model, lora_config)
print("LoRA applied to the model.")
model.print_trainable_parameters() # 打印可训练参数信息


################################################
# --------------- 6.训练配置 --------------------
################################################

import wandb

wandb.init(project="finetune_dsr1-4")


print("Configuring Training Arguments...")
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,       # 根据你的显存调整，int8 下可以适当增大
    gradient_accumulation_steps=8,       # 有效 batch size = batch_size * accumulation_steps
    optim = "adamw_8bit",                # 使用 bitsandbytes 提供的 8-bit 优化器以节省显存
    # optim="adamw_torch",               # 或者使用标准的 AdamW
    save_steps=500,                      # 每 N 步保存一次 checkpoint (LoRA adapter)
    logging_steps=2,                    # 每 N 步记录一次日志
    learning_rate=1e-4,                  # LoRA 常用的学习率范围 (1e-4 到 5e-4)
    num_train_epochs=1,                  # 训练轮数，根据数据集大小调整
    max_grad_norm=0.3,                   # 梯度裁剪阈值
    lr_scheduler_type="cosine",          # 学习率调度器类型
    weight_decay = 0.01,
    warmup_ratio=0.03,                   # 预热比例
    fp16=False,                          # 如果 GPU 支持 bf16，设为 True 更好。int8量化时混合精度类型要小心设置
    bf16=True,                           # 如果 GPU 支持 bf16 (如 A100, H100), 推荐使用
    # 如果都不支持，注释掉 fp16 和 bf16
    # group_by_length=True,              # 可选：将相似长度的序列分组，提高效率 (SFTTrainer 配合 packing 可能不需要)
    # report_to="tensorboard",             # 将日志报告给 tensorboard (需要安装 tensorboard)
    report_to="wandb",                 # 或者 wandb (需要安装 wandb 并登录)  
)

print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,                         # 基础模型 (已量化)
    tokenizer=tokenizer,
    args=training_arguments,
    train_dataset=dataset,               # 格式化后的数据集
    peft_config=lora_config,             # LoRA 配置
    dataset_text_field="text",           # 数据集中包含格式化后文本的列名
    dataset_num_proc = 2,
    max_seq_length=2048,                 # 最大序列长度，根据模型和显存调整
    packing=False,                        # 推荐开启：将短序列打包成长序列，提高效率
    # formatting_func=formatting_prompts_func, # 如果你想让 Trainer 处理格式化，而不是预先 .map()
                                         # 但需要调整 formatting_func 的签名以处理批次
)

# --- 10. 开始训练 ---
print("Starting training...")
trainer.train()
print("Training finished.")


### 用nohup accelerate启动训练
# nohup accelerate launch train_on_single_gpu.py > training.log 2>&1 &

### 停止训练
# ps aux | grep "accelerate train_on_single_gpu.py"
# kill 进程
