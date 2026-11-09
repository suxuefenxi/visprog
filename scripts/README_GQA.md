# VisProg GQA 批量实验指南（服务器）

## 1. 文件说明

| 文件 | 作用 |
|---|---|
| `scripts/download_gqa.py` | 下载 GQA testdev-balanced（12578 题 / 398 图），随机抽样并保存 |
| `scripts/run_gqa_batch.py` | 批量生成程序、执行、统计准确率，支持断点续跑 |

数据来源：HuggingFace `lmms-lab/GQA`。默认抽样 500 题。

## 2. 操作步骤

### 2.1 准备环境（服务器，需 GPU）
```bash
git clone <你的仓库地址> visprog && cd visprog
conda env create -f environment.yaml      # 或自行建 python3.10 环境
conda activate visprog
```
注意：
- `environment.yaml` 已改为 `openai>=1.55.3`，并加入 `datasets`；同时固定 `huggingface_hub==0.25.2`，因为 `diffusers==0.14.0` 依赖新版已删除的 `cached_download`。
- 引擎在 import 时就需要 `face_detection`、`augly`、`diffusers`，GQA 用不到它们，但必须能装上。
- 需要安装带 CUDA 的 PyTorch，否则 BLIP/OWL-ViT 会在 CPU 上运行，速度很慢。

### 2.2 下载数据
```bash
# 国内服务器可加镜像
python scripts/download_gqa.py --out data/gqa --num 500 --seed 0 --hf-endpoint https://hf-mirror.com
```
`--num 0` 表示全量。输出为 `data/gqa/questions.jsonl` 和 `data/gqa/images/*.jpg`。

### 2.3 配置 LLM
原 `text-davinci-003` 已下线，`ProgramGenerator` 已改用 chat 接口，可接任意 OpenAI 兼容服务：
```bash
export OPENAI_MODEL=gpt-3.5-turbo                  # 默认模型，也可用 --model 覆盖
```
```bash
export OPENAI_API_KEY=sk-xxxx
export OPENAI_BASE_URL=https://api.openai.com/v1   # 用 DeepSeek 等服务时改成其地址
# 国内服务器首次下载 BLIP / OWL-ViT 权重时：
export HF_ENDPOINT=https://hf-mirror.com
```

### 2.4 冒烟测试（先跑 5 题）
```bash
python scripts/run_gqa_batch.py --limit 5 --run-name smoke --model gpt-3.5-turbo
```
确认能正常出结果（首次运行会下载模型权重）。查看 `outputs/gqa/smoke/results.jsonl`，检查每题的 `program`、`pred`、`error`。

### 2.5 正式批量运行
```bash
nohup python scripts/run_gqa_batch.py --run-name run1 --model gpt-3.5-turbo > run1.log 2>&1 &
tail -f run1.log
```
- 每题结果会立刻写入 `outputs/gqa/run1/results.jsonl`。
- 中断后重新执行同一命令会自动跳过已完成的题（断点续跑）。
- 结束后自动打印并保存 `outputs/gqa/run1/summary.json`，包含总准确率、按 `structural` 和 `semantic` 分类的准确率、程序执行失败数。
- 只想重新统计时：`python scripts/run_gqa_batch.py --run-name run1 --eval-only`

### 2.6 常用参数
| 参数 | 说明 |
|---|---|
| `--limit N` | 只跑前 N 题 |
| `--model` | LLM 名称，如 `gpt-4o-mini`、`deepseek-chat` |
| `--prompt-method all/random` | 使用全部 prompt 示例（默认）或随机抽取 `--num-prompts` 个 |
| `--temperature` | 默认 0.7；想要可复现可设为 0 |
| `--max-retries` | 生成或执行失败时的重试次数，默认 3 |

## 3. 注意事项
- 准确率为小写、去空格后的 exact match（GQA 标准）。程序执行最终失败时，预测为空串，按错误计算。
- `prompts/gqa.py` 中有若干示例的笔误（如 `RESULT(var=ANSWER)`、`if {ANSWER0} > 0 and else`，以及第 17、18 个示例之间缺逗号）。这些会影响效果，需要复现论文时可以先修掉。
- 图片在推理前会缩放到最长边 640，以节省显存。
- 费用与耗时：每题约 1 次 LLM 调用（prompt 约 2–3k tokens）加 GPU 推理，500 题通常在数十分钟内完成，具体取决于 GPU 和 API 延迟。
- 显存：BLIP-VQA-large 和 OWL-ViT-large 同时加载，建议显存不少于 12GB。
