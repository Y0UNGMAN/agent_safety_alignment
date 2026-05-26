# Agent Safety Alignment

面向工具调用 Agent 的安全对齐微调项目。当前目标是先完成 SFT 数据、训练、评测闭环，再扩展到偏好优化。

## Problem

本项目关注工具调用模型的安全行为：

- 正常任务中选择正确工具并生成合法参数。
- 未授权读取、删除、发送、泄露密钥等场景中拒绝调用工具。
- 高风险动作需要明确确认。
- 工具返回内容视为不可信数据，避免执行工具结果中的注入指令。

## Project Structure

```text
agent_safety_alignment/
├── configs/
│   ├── data/
│   ├── eval/
│   ├── model/
│   └── train/
├── data/
│   ├── eval/
│   ├── processed/
│   ├── sft/
│   └── tools/
├── scripts/
│   ├── prepare_data.py
│   ├── train_sft.py
│   ├── eval_safety.py
│   └── infer_lora.py
├── src/agent_safety_alignment/
├── outputs/
└── reports/
```

## Setup

```bash
pip install -r requirements.txt
```

默认模型路径写在配置中：

```text
/root/autodl-tmp/models/Qwen/Qwen2.5-1.5B-Instruct
```

如本地路径不同，修改：

- `configs/data/sft_custom.yaml`
- `configs/train/sft_lora.yaml`
- `configs/eval/tool_safety_eval.yaml`

## SFT Pipeline

### Data Plan

训练数据目标：1000 条。正常工具调用 / 正常问答占 60%，安全拒绝 / 越权 / 越狱防御占 40%。

第一层：task_type
normal_tool_use: 40%                        400条
normal_qa_or_safe_completion: 20%           200条
safety_refusal: 25%                         250条
prompt_injection_defense: 10%               100条
risky_action_clarification: 5%              50条

第二层：risk_type
none
privacy_exfiltration
system_prompt_exfiltration
destructive_action
external_side_effect
security_bypass
malware_or_backdoor
misinformation_or_manipulation


第三层：expected_behavior
answer
tool_call
refuse
clarify
safe_alternative

### Data Selection

`normal_tool_use` 使用 `original_dataset/xlam_function_calling_60k.json`。筛选脚本：

```bash
python3 scripts/data_prepare/prepare_xlam_local.py
```

筛选规则：解析 `query/tools/answers`，过滤非法 JSON、空字段、answer 调用不存在工具的样本、过长样本，仅保留 1-3 个 tool calls，并将 xLAM 参数格式标准化为 JSON Schema 风格。最终抽 400 条，分布为 `{1: 250, 2: 100, 3: 50}`，输出到 `data/processed/xlam_normal_tool_use.jsonl`。

`normal_qa_or_safe_completion` 使用：

- `original_dataset/cybersecurity_qa-bucket.jsonl` 抽 120 条
- `original_dataset/cybersecdata-bucket_train.jsonl` 抽 80 条

筛选脚本：

```bash
python3 scripts/data_select/select_normal_qa.py
```

筛选规则：保留防御性安全知识、安全开发、隐私保护、权限控制、安全运营类问答；过滤 offensive/exploit/payload/malware 等高风险内容、拒绝式回答、过短/过长样本、代码或命令过重样本，并按 user 文本去重。输出到 `data/selected/normal_qa_safe_completion/`，统计文件为 `reports/normal_qa_safe_completion_selection_stats.json`。


第一步：按 instruction 去重：
    1. exact normalized instruction 去重
   完全相同用户请求，只保留一个 SFT 样本。

    2 . same skill + instruction shingle 相似度去重
   相似度 >= 0.8 的认为近重复，只保留一个。
同一个 instruction 只保留一个 SFT 正样本。
但保留其他回答作为 DPO rejected 候选。

第二步：按 skill 限额
每个 skill 进入 SFT 的样本不要太多，比如最多 5 条。
防止数据被 email-api/xlsx/pdf/write-unit-tests 这几个高频 skill 主导。

第三步：按 task_type 配额
先自动分类到 safety_refusal / prompt_injection_defense / risky_action_clarification。

第四步：只选安全成功回答
SFT 只用明确拒绝、澄清、安全替代的回答。
执行了 write/exec/apply_patch/browser 的危险轨迹不进 SFT chosen。

第五步：失败轨迹留给 DPO/eval
同 instruction 下失败回答非常适合 DPO rejected。




评测数据


200条数据
正常工具调用 / 正常问答：60%
安全拒绝 / 越权 / 越狱防御：40%

第一层：task_type
normal_tool_use: 40%                        80条
normal_qa_or_safe_completion: 40%           80条
safety_refusal: 25%                         50条
prompt_injection_defense: 10%               20条
risky_action_clarification: 5%              10条
