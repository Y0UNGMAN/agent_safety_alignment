# -*- coding: utf-8 -*-
# 导入必要的库
import os  # 用于操作系统交互，例如检查环境变量
import argparse # 用于解析命令行参数
import torch # PyTorch 核心库
from datasets import load_dataset # Hugging Face Datasets 库，用于加载数据集
from transformers import (
    AutoModelForCausalLM, # 自动加载因果语言模型
    AutoTokenizer,       # 自动加载分词器
    BitsAndBytesConfig,  # 配置 bitsandbytes 量化参数
    TrainingArguments,   # 配置训练过程的参数
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training # PEFT 库，用于 LoRA 微调
from trl import SFTTrainer # TRL 库，专门用于监督微调的 Trainer
import wandb # 用于实验跟踪和可视化 (Weights & Biases)
from functools import partial # 用于包装函数，方便传递额外参数

# 定义一个固定的中文训练提示模板
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

# --- 命令行参数解析函数 ---
def parse_args():
    parser = argparse.ArgumentParser(description="使用 SFTTrainer、LoRA、int8 和 DeepSpeed 微调模型")

    # 模型和数据路径相关参数
    parser.add_argument("--model_id_or_path", type=str, default="DeepSeek-R1-Distill-Qwen-7B", help="预训练模型的 ID 或本地路径")
    parser.add_argument("--dataset_path", type=str, default="dataset/medical_o1_sft_Chinese.json", help="JSON 数据集文件的路径")
    parser.add_argument("--output_dir", type=str, default="sft_lora_model_int8_ds_output", help="保存训练好的 LoRA 适配器和日志的目录")

    # 量化相关参数 (使用 store_true/store_false 来创建布尔开关)
    parser.add_argument("--load_in_8bit", action='store_true', default=True, help="以 8-bit 精度加载模型")
    parser.add_argument("--no_load_in_8bit", action='store_false', dest='load_in_8bit', help="不以 8-bit 精度加载模型")

    # LoRA 配置参数
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA 的秩 (rank)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA 的 alpha 缩放因子")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA 层的 dropout 概率")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", help="要应用 LoRA 的目标模块名称 (逗号分隔)")

    # 训练超参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="训练时每个 GPU 的批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数 (梯度更新前累积多少个 micro-batch)")
    parser.add_argument("--optimizer", type=str, default="adamw_8bit", choices=["adamw_torch", "adamw_8bit"], help="使用的优化器 (adamw_8bit 使用 bitsandbytes 优化版本)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="训练的总轮数 (epochs)")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="梯度裁剪的阈值")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器的类型 (例如 'linear', 'cosine')")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="学习率预热 (warmup) 阶段占总训练步数的比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减系数")
    parser.add_argument("--save_steps", type=int, default=500, help="每隔多少步保存一次检查点 (checkpoint)")
    parser.add_argument("--logging_steps", type=int, default=10, help="每隔多少步记录一次训练日志")
    parser.add_argument("--bf16", action='store_true', default=True, help="使用 bfloat16 混合精度 (如果 GPU 支持且未使用 DeepSpeed 的 fp16)")
    parser.add_argument("--no_bf16", action='store_false', dest='bf16', help="不使用 bfloat16 精度")
    parser.add_argument("--gradient_checkpointing", action='store_true', default=True, help="启用梯度检查点以节省显存")
    parser.add_argument("--no_gradient_checkpointing", action='store_false', dest='gradient_checkpointing', help="禁用梯度检查点")

    # SFTTrainer 特定参数
    parser.add_argument("--max_seq_length", type=int, default=2048, help="输入序列的最大长度")
    parser.add_argument("--packing", action='store_true', default=False, help="使用 SFTTrainer 的 packing 技术 (将短序列打包，提高效率)")
    parser.add_argument("--no_packing", action='store_false', dest='packing', help="不使用 packing")

    # DeepSpeed 参数
    # 这个参数通常由 `accelerate launch` 根据 `--deepspeed_config_file` 自动填充
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed 配置文件的路径 (例如 ds_config.json)")

    # 日志记录参数
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "tensorboard", "none"], help="将日志报告给 W&B 或 TensorBoard，或不报告")
    parser.add_argument("--wandb_project", type=str, default="finetune_dsr1-ds", help="Wandb 项目的名称")

    # 解析参数
    args = parser.parse_args()
    # 将逗号分隔的目标模块字符串转换为列表
    args.lora_target_modules = [module.strip() for module in args.lora_target_modules.split(',')]

    return args

# --- 数据集格式化函数 ---
def formatting_prompts_func(examples, tokenizer):
    """根据模板格式化数据集的 'text' 字段，用于 SFTTrainer"""
    inputs = examples['Question']    # 获取 'Question' 列的数据
    cots = examples["Complex_CoT"] # 获取 'Complex_CoT' 列的数据
    outputs = examples["Response"]   # 获取 'Response' 列的数据
    texts = []                       # 存储格式化后的文本
    EOS_TOKEN = tokenizer.eos_token # 获取分词器的结束符标记
    # 遍历每条样本
    for input_text, cot, output in zip(inputs, cots, outputs):
        # 使用预定义的模板格式化文本，并在末尾添加结束符
        text = train_prompt_style_zh.format(input_text, cot, output) + EOS_TOKEN
        texts.append(text)
    # 返回包含新 'text' 列的字典，这是 SFTTrainer 需要的格式
    return {"text": texts}

# --- 主函数 ---
def main():
    # 1. 解析命令行参数
    args = parse_args()

    # 2. 加载分词器 (Tokenizer)
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id_or_path, trust_remote_code=True)

    # 重要：设置 Padding Token (填充标记)
    # SFTTrainer 和模型都需要一个填充标记。对于很多自回归模型，通常将其设为结束标记 (eos_token)
    if tokenizer.pad_token is None:
        print("分词器没有 pad_token，将其设置为 eos_token。")
        tokenizer.pad_token = tokenizer.eos_token
        # 注意：模型配置的 pad_token_id 通常由 Trainer 或模型加载时自动处理

    # 3. 加载数据集
    print("加载数据集...")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train") # 从 JSON 文件加载训练集
    print(f"数据集加载完成，共 {len(dataset)} 条样本。")
    print(f"数据集包含的列: {dataset.column_names}")

    # 4. 处理数据集 (格式化)
    print("格式化数据集...")
    # 使用 functools.partial 将 tokenizer 作为固定参数传给 map 函数
    formatting_func = partial(formatting_prompts_func, tokenizer=tokenizer)
    # 使用 .map() 方法应用格式化函数，`batched=True` 表示批量处理以提高效率
    dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names) # remove_columns 可选，移除旧列只保留 'text'
    print("数据集格式化完成:", dataset)
    if len(dataset) > 0:
         print("第一条样本预览:", dataset[0]['text'][:300] + "...") # 打印部分格式化后的文本
    else:
         print("错误：格式化后数据集为空！")
         return # 如果数据集为空则退出

    # 5. 配置量化 (BitsAndBytes)
    bnb_config = None # 默认为不量化
    if args.load_in_8bit:
        print("配置 BitsAndBytes 进行 int8 量化...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True, # 启用 8-bit 量化
        )
    else:
        print("不进行量化，加载原始模型。")

    # 6. 加载基础模型
    print(f"加载基础模型 '{args.model_id_or_path}'...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id_or_path,
        # quantization_config=bnb_config, # 传入量化配置 (如果启用)
        # device_map="auto", # **移除**：在使用 accelerate 进行分布式训练时，不需要手动指定 device_map，accelerate 会自动处理设备分配
        trust_remote_code=True # 信任模型仓库中的自定义代码 (对于某些模型是必需的)
    )
    print("基础模型加载完成。")

    # 7. 配置 LoRA
    print("配置 LoRA...")
    if args.load_in_8bit:
        print("为 K-bit (int8) 训练准备模型...")
        # 这个函数会做一些准备工作，比如确保某些层是可训练的，并处理梯度检查点兼容性
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    # 创建 LoRA 配置对象
    lora_config = LoraConfig(
        r=args.lora_r,                   # LoRA 秩
        lora_alpha=args.lora_alpha,          # LoRA alpha 缩放
        target_modules=args.lora_target_modules, # 要应用 LoRA 的层列表
        lora_dropout=args.lora_dropout,     # LoRA dropout
        bias="none",                     # 通常 LoRA 不训练 bias
        task_type="CAUSAL_LM"            # 任务类型，对于 GPT 类模型是因果语言模型
    )

    print("将 LoRA 配置应用到模型...")
    # 使用 get_peft_model 将 LoRA 适配器层添加到基础模型上
    model = get_peft_model(model, lora_config)
    print("LoRA 应用完成。")
    # 打印模型中可训练参数的数量和比例，检查 LoRA 是否配置成功
    model.print_trainable_parameters()

    # 8. 配置训练参数 (TrainingArguments)
    print("配置训练参数...")

    # --- Wandb 初始化 ---
    # 只在主进程 (rank 0) 上初始化 wandb，避免多进程重复初始化和记录
    # accelerate 通常会自动处理，但显式检查更安全 (LOCAL_RANK 是 accelerate 设置的环境变量)
    if (args.report_to == "wandb") and (os.environ.get("LOCAL_RANK", "0") == "0"):
         # 使用命令行参数或默认值初始化 wandb
         wandb.init(project=args.wandb_project, config=vars(args)) # config=vars(args) 会记录所有命令行参数

    # 创建 TrainingArguments 对象
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,                             # 输出目录
        per_device_train_batch_size=args.per_device_train_batch_size, # 每个设备的批大小
        gradient_accumulation_steps=args.gradient_accumulation_steps, # 梯度累积步数
        optim=args.optimizer,                                   # 优化器类型
        save_steps=args.save_steps,                             # 保存检查点的步数间隔
        logging_steps=args.logging_steps,                       # 记录日志的步数间隔
        learning_rate=args.learning_rate,                       # 学习率
        num_train_epochs=args.num_train_epochs,                 # 训练轮数
        max_grad_norm=args.max_grad_norm,                       # 梯度裁剪阈值
        lr_scheduler_type=args.lr_scheduler_type,               # 学习率调度器类型
        weight_decay=args.weight_decay,                         # 权重衰减
        warmup_ratio=args.warmup_ratio,                         # 预热比例
        fp16=True,                                             
                                                                # DeepSpeed 启用时，由 DeepSpeed 配置决定精度
        gradient_checkpointing=args.gradient_checkpointing,     # 是否启用梯度检查点
        report_to=args.report_to if os.environ.get("LOCAL_RANK", "0") == "0" else "none", # 只在主进程报告日志
        deepspeed=args.deepspeed,                               # **关键**：传入 DeepSpeed 配置文件的路径
        save_total_limit=3,                                     # 可选：限制保存的检查点总数
        logging_first_step=True,                                # 可选：在第一个 step 就记录日志
    )

    # 9. 初始化 SFTTrainer
    print("初始化 SFTTrainer...")
    trainer = SFTTrainer(
        model=model,                     # 传入我们配置好的 PeftModel (基础模型 + LoRA 适配器)
        tokenizer=tokenizer,             # 传入分词器
        args=training_arguments,         # 传入训练参数
        train_dataset=dataset,           # 传入格式化后的训练数据集
        peft_config=lora_config,         # 传入 LoRA 配置 (虽然已应用到 model，但 Trainer 可能也需要)
        dataset_text_field="text",       # 指定数据集中包含训练文本的列名
        max_seq_length=args.max_seq_length, # 最大序列长度
        packing=args.packing,            # 是否启用 packing
        # dataset_num_proc=4,            # 可选：用于数据预处理的进程数，根据 CPU 和数据集大小调整
    )

    # 10. 开始训练
    print("开始训练...")
    # 调用 train() 方法启动训练过程
    train_result = trainer.train()
    print("训练完成。")

    # 11. 保存最终模型和状态
    # 训练过程中会根据 save_steps 保存 LoRA 适配器。这里显式保存最后的状态。
    print(f"将最终的 LoRA 适配器保存到 {args.output_dir}")
    trainer.save_model(args.output_dir) # 这会保存 adapter_model.bin 和 adapter_config.json

    # 记录训练指标
    metrics = train_result.metrics
    # 只在主进程记录和保存指标
    if trainer.is_world_process_zero():
        print("记录训练指标...")
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state() # 保存 Trainer 的状态 (包括优化器状态、随机种子等)
        print("训练指标和最终状态已保存。")

    # 结束 wandb 运行 (只在主进程)
    if (args.report_to == "wandb") and (os.environ.get("LOCAL_RANK", "0") == "0"):
        wandb.finish()

# Python 脚本的入口点
if __name__ == "__main__":
    main() # 执行主函数