# KV Cache管理架构演进：从连续分配到统一混合内存架构

> **源链接**：[https://developer.aliyun.com/article/1714337](https://developer.aliyun.com/article/1714337)  
> **作者**：Deephub / Luv Bansal  
> **发布日期**：2026-03-03  
> **摘要**：本文系统梳理 KV Cache 管理演进的 5 个时代（从无到统一内存架构），剖析 vLLM、SGLang、TensorRT-LLM 等框架在各阶段的技术取舍与实践效果，涵盖连续缓存、PagedAttention、异构/分布式/统一混合架构等关键突破，为不同场景（文本、多模态、长上下文、混合模型）选择最优方案。

## 一、 背景：Prefill、Decode 与 KV Cache

在生产环境部署过 LLM 的人都知道模型权重只是问题的一半，另一半是 **KV Cache**：存储注意力状态的运行时内存，让模型在生成 token 时不必从头开始重算。能不能管好这块内存决定了系统是一个卡顿的 demo 还是一个可用的推理服务。

LLM 推理分为两个阶段：
1. **Prefill 阶段**：并行处理全部输入 token，在每个注意力层为每个 token 计算 Key 和 Value 向量，属于**计算密集型**，GPU 并行度越高越好。
2. **Decode 阶段**：以自回归方式逐 token 生成，每个新 token 都要对先前所有 Key-Value 对做注意力计算；GPU 大部分时间花在从 HBM 读取 KV Cache 而非运算上，**瓶颈在内存带宽**。

KV Cache 的作用就是把已经算过的 Key 和 Value 向量缓存下来，避免每个 decode 步骤重复计算。没有它每生成一个 token 就得对整个序列重跑一遍注意力，推理速度完全无法接受。

以 **Llama-3–70B、8K 上下文** 为例：
```text
KV cache per token = 2 (K+V) x 80 layers x 8 KV heads x 128 head_dim x 2 bytes (FP16)  
                   = 2 x 80 x 8 x 128 x 2 = 327,680 bytes ≈ 320 KB per token  

For 8K tokens: 320 KB x 8,192 = 2.56 GB per request  
For 32 concurrent requests: 2.56 GB x 32 = 81.9 GB
```

**81.9 GB**：一块 A100 80GB 的全部显存都装不下，留给模型权重的空间是零。KV Cache 管理极其重要正是因为这一点。

## 二、 KV Cache 演进的 5 个时代

### Era 0：Pre-GenAI（2017 年之前）
- **背景**：Transformer 出现之前，深度学习的主力是 ResNet、YOLO、VGG、Inception 等无状态前馈架构。
- **特点**：每次推理独立处理一个输入，步骤之间没有任何持久状态，KV Cache 的概念无从谈起。
- **推理框架**：ONNX Runtime、TensorRT 等早期的无状态推理框架。

### Era 1：连续 KV Cache（2017 年）
- **背景**：Transformer 原始论文（2017）带来了自注意力机制，带来了在 decode 步骤之间缓存 Key 和 Value 张量的需求。
- **实现方式**：早期推理引擎（如 HuggingFace Transformers）为每个请求预分配一个 `max_seq_len` 大小的连续张量：
```text
Size per request = 2 × num_layers × num_heads × head_dim × max_seq_len
```
- **优点**：实现简单，相比每步重算注意力有巨大的速度提升。
- **代价与问题**：
  - 内存占用按 `max_seq_len * batch_size` 线性增长而非跟随实际序列长度；
  - 产生严重的**内部碎片**（已分配显存中只有 20–38% 真正存储了有用 token）；
  - 并发 Batch 受限，且请求之间无法共享内存。

### Era 2：PagedAttention（2023 年）
- **突破**：UC Berkeley vLLM 团队借鉴了操作系统的**带分页虚拟内存**机制。
- **机制**：
  - 将 KV Cache 切分为固定大小的页（Block），随着序列增长**按需动态分配**。
  - 通过 **Block Table** 将逻辑页映射到物理内存。
- **优势**：
  - 内存浪费率降至 4% 以下，大幅提升并发吞吐。
  - **前缀缓存（Prefix Caching）**：SGLang 的 **RadixAttention** 基于此实现，多轮对话/RAG 场景中共享前缀的 KV Block 可直接复用，显著提升吞吐。
- **取舍**：注意力 Kernel 访问非连续内存变复杂，Block 大小需调优，默认假设 KV Cache 也是同构的（每层大小一致）。
- **实践对比（vLLM vs SGLang）**：
  - **vLLM**：Block 级别的哈希前缀匹配，方案简洁，标准聊天场景表现良好。
  - **SGLang**：RadixAttention 树（基数树）维护 LRU 缓存，在复杂的 Agent / 思维树（Tree-of-Thought）多调用场景下缓存命中率更高。

### Era 3：异构 KV Cache（2024 年）
- **背景**：2024 年模型架构与优化技术快速分化，推理系统需要管理形状、生命周期、访问模式各异的多种缓存状态。
- **常见异构缓存类型**：
  1. **投机解码 (Speculative Decoding)**：草稿模型与目标模型各自维护独立 KV Cache。
  2. **多模态模型 (VLM)**：视觉编码器产生大型图像嵌入（如 QwenVL, InternVL），尺寸与文本 KV Cache 不同。
  3. **量化 KV Cache**：FP8 / INT4 等低精度格式存储，需额外维护 scaling factor。
  4. **滑动窗口注意力 (SWA)**：仅保留 `window_size` 内的 token，需动态淘汰过期 KV。
  5. **SSM / Mamba 状态**：使用循环状态代替注意力，固定大小向量，无法在 token 粒度共享或回滚。
  6. **混合架构模型**：单个模型融合多种 Layer：
     - 滑动窗口 + 全注意力（Gemma 2/3, Ministral）
     - Mamba + 全注意力（Jamba, Bamba）
     - 局部分块 + 全注意力（Llama 4）
- **挑战**：Jenga 论文表明，Llama 3.2 11B Vision 若统一传统管理，内存浪费达 **79.6%**。独立分配器带来严重的内存碎片与扩展难题。

### Era 4：分布式 KV Cache（2025+）
- **背景**：单 GPU 甚至单节点无法容纳超长上下文与极高并发，KV Cache 管理扩展到多节点、数据中心级别。
- **核心技术路径**：
  1. **解耦推理 (Disaggregated Inference)**：
     - **DistServe**：将 Prefill（计算密集型）和 Decode（内存带宽密集型）部署到不同 GPU 实例，吞吐提升 4.48 倍。
     - **Encoder Disaggregation (vLLM)**：将视觉编码器分离为独立服务，Goodput 提升 2–2.5 倍。
  2. **KV Cache 感知负载均衡 (KV-Aware Routing)**：
     - **NVIDIA Dynamo**：路由按全局 KV 缓存视图将请求发送给已存在前缀缓存的节点，最大化前缀命中率。
  3. **分层 KV Cache (Tiered KV Cache)**：
     - **Mooncake (Moonshot AI)**：以 KV Cache 为中心，冷 KV 页从 GPU HBM 溢出到 CPU DRAM 或 SSD，通过传输与 GPU 计算重叠隐藏延迟。长上下文吞吐最高提升 525%。
- **困难**：复杂网络传输（RoCE/NIXL）、故障转移、节点扩缩容与状态同步。

### Era 5：统一混合 KV Cache（2025+）
- **核心方向**：构建统一内存系统，让异构 KV 类型共享同一个内存池，实现极佳的**功能可组合性 (Combinability)**。
- **代表性突破**：
  1. **Jenga (大页 + LCM 尺寸对齐)**：
     - 提出两级分配器，取不同嵌入尺寸的最小公倍数（LCM）作为“大页”尺寸（如 `LCM(256, 384) = 768` 字节），消除了不同形状 KV 混合时的碎片。
     - GPU 内存利用率改善最高 79.6%，吞吐提升最高 4.92 倍。
  2. **SGLang (CUDA 虚拟内存 API)**：
     - 使用 CUDA Virtual Memory API 动态重映射设备内存，实现逻辑连续、物理分散。
     - 弹性内存池在运行时动态调整不同池（如 Mamba 池与 KV 缓存池）的分配比例。

## 三、 5 个时代对比总结表

| 时代 | 代表技术 / 框架 | 核心特征 | 内存利用率 / 优势 | 主要挑战 / 取舍 |
| :--- | :--- | :--- | :--- | :--- |
| **Era 0: Pre-GenAI** (< 2017) | ONNX Runtime, TensorRT | 无状态推理，无 KV 缓存 | 无运行时 KV 状态开销 | 无法支持自回归生成 |
| **Era 1: 连续缓存** (2017) | Early Transformers | 预分配固定 max_seq_len 连续内存 | 实现极简 | 严重内部碎片（浪费 60%-80%），无法共享 |
| **Era 2: PagedAttention** (2023) | vLLM, SGLang, TRT-LLM | 虚拟分页管理，前缀缓存 | 碎片率 < 4%，高并发共享前缀 | 非连续访问 Kernel 复杂，默认假设层同构 |
| **Era 3: 异构缓存** (2024) | VLM, SWA, Speculative, Mamba | 针对不同层/模态/量化管理多种形状缓存 | 支持复杂模型结构 | 独立分配器引发二次碎片，扩展脆弱 |
| **Era 4: 分布式缓存** (2025+) | DistServe, Mooncake, Dynamo | PD 解耦、KV 感知路由、HBM-DRAM-SSD 分层 | 长上下文吞吐提升数倍 | 跨节点网络传输瓶颈，故障恢复复杂 |
| **Era 5: 统一混合内存** (2025+) | Jenga, SGLang CVM | 大页 LCM 对齐、CUDA 虚拟内存重映射 | 极高利用率与吞吐，优化可完美组合 | 架构需大幅重构，工程难度高 |

## 四、 场景选型指南

1. **标准文本 LLM 服务（聊天/对话/RAG）**：
   - 采用 **Era 2 (PagedAttention)** 即可。使用 vLLM 或 SGLang，并启用前缀缓存（Prefix Caching）。
2. **多模态模型 (VLM)**：
   - 需考虑 **Era 3** 对图像嵌入的处理。高并发场景建议评估 vLLM 的 **Encoder Disaggregation (Era 4)**。
3. **混合架构模型（Gemma 3, Jamba, Llama 4）**：
   - 依赖 **Era 5** 统一混合内存，首选支持 SGLang (CUDA Virtual Memory) 或 Jenga 分配器机制的引擎。
4. **大规模高吞吐生产集群**：
   - 必须应用 **Era 4** 的 PD 解耦（Prefill/Decode Disaggregation）与 KV 感知路由（如 NVIDIA Dynamo, Mooncake）。
5. **超长上下文负载（100K+ Tokens）**：
   - 必须依赖 **Era 4 分层 KV Cache**（GPU HBM 溢出到 CPU DRAM / NVMe SSD）。

## 五、 结语

KV Cache 是大语言模型推理的真正性能瓶颈。KV Cache 管理的演进轨迹与操作系统内存管理历史惊人地相似：**从连续分配到虚拟内存分页，再到分布式共享内存**。操作系统用 40 年走完的路，KV Cache 管理在 8 年内迅速重走了一遍。深刻理解这一演进脉络，是构建高性能 LLM 推理基础设施的基石。

---
*原文链接：https://developer.aliyun.com/article/1714337*
