# 7. TensorRT-LLM 快速入门指南与架构解析

## 1. 框架概述

**TensorRT-LLM** 是 NVIDIA 官方专门为大语言模型（LLM）高性能推理打造的高阶 C++/Python 开源加速库。它建立在 NVIDIA TensorRT 深度学习编译器之上，集成了当前业界最前沿的 LLM 推理优化技术（如 PagedKV Cache、In-Flight Batching / Continuous Batching、Tensor/Pipeline Parallelism 跨卡并行、FP8/INT4 极佳量化等），专为在 NVIDIA GPUs（如 H100, A100, L40S, RTX 4090 等）上实现超高吞吐和极致低延迟而设计。

* **官方文档**: [TensorRT-LLM Quick Start Guide](https://nvidia.github.io/TensorRT-LLM/latest/quick-start-guide.html)
* **GitHub 仓库**: [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

---

## 2. 核心工作流程（Deployment Workflow）

TensorRT-LLM 的推理流程采用 **AOT（Ahead-Of-Time，预编译）** 机制。整个部署过程包含三步标准流水线：

```mermaid
graph LR
    A[Hugging Face 模型权重] -->|Step 1: 格式转换| B[TensorRT-LLM Checkpoint]
    B -->|Step 2: trtllm-build 编译| C[硬件专属 Engine 文件]
    C -->|Step 3: trtllm-serve / API| D[OpenAI 兼容推理服务]
```

### 步骤详解：

1. **模型转换（Checkpoint Conversion）**：
   将 Hugging Face / PyTorch 原始权重转换为 TensorRT-LLM 统一的数据格式与网络拓扑。
2. **引擎编译（Engine Compilation - `trtllm-build`）**：
   使用核心 CLI 工具 `trtllm-build` 将模型编译为专属于目标 GPU（例如特定 Compute Capability 的卡）的 `.engine` 二进制文件。在此步骤中，TensorRT-LLM 会自动完成**算子融合（Kernel Fusion）**、**显存分配策略优化**、**量化模式应用**以及**张量并行/流水线并行切分**。
3. **服务部署（Model Serving - `trtllm-serve`）**：
   加载编译好的 `.engine` 引擎，启动具有高并发处理能力的 OpenAI 兼容 API 服务器或使用 Python API 直接推理。

---

## 3. 环境搭建与快速开始 (Quick Start)

### 3.1 使用 NGC 官方预构建 Docker 镜像（推荐）

为避免复杂的 CUDA、CUDNN、TensorRT 版本依赖冲突，NVIDIA 推荐直接使用 NGC 容器：

```bash
# 1. 拉取并启动最新的 TensorRT-LLM 官方镜像
docker run --rm -it --ipc=host \
  --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  nvcr.io/nvidia/tensorrt-llm/release:latest
```

---

## 4. 核心命令行工具实操详解

### 4.1 方式一：极简一键服务启动（`trtllm-serve`）

如果不需要手动调节编译细节，可以直接使用 `trtllm-serve` 启动服务器，系统会在后台自动完成模型转换与编译：

```bash
# 启动 OpenAI 格式服务（以 TinyLlama 为例）
trtllm-serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --host 0.0.0.0 --port 8000
```

服务启动后，使用标准的 `curl` 命令发送请求：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Hello! Introduce TensorRT-LLM in 3 sentences."}]
  }'
```

---

### 4.2 方式二：精细控制——AOT 编译模式（`trtllm-build`）

在大规模生产环境中，通常需要手动控制并行度与量化精度：

#### 步骤 1：转换模型权重
```bash
python3 examples/llama/convert_checkpoint.py \
    --model_dir /path/to/llama-3-8b \
    --output_dir ./tllm_checkpoint_tp2 \
    --tp_size 2 \
    --dtype float16
```

#### 步骤 2：使用 `trtllm-build` 编译引擎
```bash
trtllm-build \
    --checkpoint_dir ./tllm_checkpoint_tp2 \
    --output_dir ./engine_outputs_tp2 \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 4096 \
    --max_output_len 2048
```

* **关键编译参数**：
  * `--gemm_plugin` / `--gpt_attention_plugin`：启用 TensorRT 高性能融合 Kernel 插件（如 FMHA、PagedAttention）。
  * `--tp_size` / `--pp_size`：配置张量并行（TP）与流水线并行（PP）节点数量。
  * `--max_batch_size` & `--max_input_len`：配置编译时分配的显存上限预算。

---

### 4.3 方式三：使用 Python API 高层接口 (`LLM`)

TensorRT-LLM 提供了类似 vLLM 风格的高层 Python API，方便在脚本中直接嵌入使用：

```python
from tensorrt_llm import LLM, SamplingParams

# 初始化 LLM 实例（自动处理编译与加载）
prompts = [
    "Hello, my name is",
    "The capital of France is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 批量生成
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

---

## 5. 高级优化特性与性能压榨

TensorRT-LLM 结合 NVIDIA 显卡硬件特性提供了深度优化：

| 优化特性 | 原理与技术优势 |
| :--- | :--- |
| **In-Flight Batching** | 即 Continuous Batching（连续批处理），在 Iteration 级别混包 Prefill 与 Decode 阶段，极大消除静态 Padding 带来的 GPU 空闲浪费。 |
| **Paged Attention** | 将 KV Cache 划分为固定大小的物理 Block，通过页表逻辑动态映射，大幅降低显存碎片并提高最大 Batch Size。 |
| **FP8 / INT4 量化** | 原生支持 Hopper/Ada 架构的 FP8 (E4M3/E5M2) 硬件计算，以及 INT4-AWQ/GPTQ 权重量化，显存体积减半，吞吐提升 2~3 倍。 |
| **Custom All-Reduce Kernels** | 针对 NVLink / NVSwitch 硬件实现的专用跨卡 All-Reduce 算子，降低 Tensor Parallelism 节点间通信开销。 |

---

## 6. vLLM 与 TensorRT-LLM 架构对比

| 对比维度 | **vLLM** | **TensorRT-LLM** |
| :--- | :--- | :--- |
| **底层内核** | C++/CUDA (Custom Kernels / FlashAttention) | TensorRT C++ Engine + CUDA Plugins |
| **编译模式** | JIT (即时执行 / Eager 模式支持度好) | AOT (Ahead-Of-Time 预编译 Engine) |
| **硬件适配** | 通用 (NVIDIA / AMD / Ascend 等) | 极致绑定 NVIDIA GPUs (优化极深) |
| **易用性与灵活度**| 高（开箱即用，Python 生态友好） | 中（需要 Build 引擎，配置参数较多） |
| **延迟/吞吐上限** | 极佳 | 极致（在 NVIDIA GPU 上硬件利用率接近极限） |
