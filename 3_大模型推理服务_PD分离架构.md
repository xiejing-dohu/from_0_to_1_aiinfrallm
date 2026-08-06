# 大模型推理服务：Prefill-Decode (PD) 分离架构深度解析

> **源链接**：[https://www.toutiao.com/article/7624403746868576819/](https://www.toutiao.com/article/7624403746868576819/?wid=1785914307657)（今日头条）  
> **摘要**：大语言模型（LLM）推理服务正在从传统的 Prefill 与 Decode 混部架构（Colocation）向 **Prefill-Decode (PD) 分离架构** 快速演进。本文深度剖析 PD 分离架构的设计动机、系统拓扑、KV Cache 跨节点 RDMA 传输机制、全局路由调度（KV-Aware Routing）以及工业级代表系统（DistServe, Mooncake, vLLM），为大规模高性能 LLM 推理集群建设提供全面技术参考。

---

## 一、 为什么需要 PD 分离架构？

在大语言模型自回归推理中，存在两个阶段且两者的计算与内存特性完全相反：

1. **Prefill（首字生成阶段）**：
   - **计算特征**：**计算密集型 (Compute-bound)**。对 Prompt 输入的所有 Token 进行并行矩阵乘法计算。
   - **关键指标**：**TTFT (Time To First Token，首字延迟)**。
   - **硬件偏好**：需要极高算力 (FLOPs) 的 GPU（如 H100/A100），适合大 Tensor Parallelism (TP) 规格。

2. **Decode（逐字生成阶段）**：
   - **计算特征**：**内存带宽密集型 (Memory-bound)**。每生成一个新 Token 都要从 HBM 读取全量权重与先前保存的 KV Cache。
   - **关键指标**：**TTPT (Time Per Output Token，字间延迟)**。
   - **硬件偏好**：需要极高 HBM 内存带宽与大显存容量，适合大 Data Parallelism (DP) 或 Pipeline Parallelism (PP)。

### 传统共置（Colocation）混部架构的致命痛点：

```text
传统混部模式 (Prefill & Decode 共享同一 GPU)
+-------------------------------------------------------------+
| GPU 实例                                                    |
|                                                             |
|  请求 A (Prefill): 占用大量算力与 GEMM 计算单元 ────┐        |
|                                                     │ 相互  |
|  请求 B (Decode) : 内存带宽受限，等待 GEMM 完成 ───┴ 干扰  |
+-------------------------------------------------------------+
导致痛点：
1. 抢占导致长尾延迟：高算力的 Prefill 打断正在 Decode 的请求，导致 TTPT 出现剧烈抖动 (Jitter)。
2. 显存碎片与分配冲突：Prefill 阶段突发的长 Prompt 容易挤爆 Decode 运行中的 KV 显存，引发抢占与 OOM。
3. SLO 优化脱节：无法独立针对 TTFT（首字）与 TTPT（字间）设定不同的扩缩容与调度策略。
```

---

## 二、 PD 分离架构核心原理解析

### 1. 系统架构拓扑

PD 分离架构将集群中的计算节点划分为两个独立的硬件资源池，由全局智能路由器（Router）统一调度：

```text
                        +-----------------------+
                        |  用户 API 请求入口    |
                        +-----------+-----------+
                                    |
                                    v
                        +-----------------------+
                        |  全局 Router / 调度器 |
                        +----+-------------+----+
                             |             |
       1. 派发 Prompt        |             |  3. 关联 Decode
       及计算 KV Cache       v             v     接收 KV Cache
              +------------------+     +------------------+
              | Prefill 节点池   |     | Decode 节点池    |
              | (Compute Cluster)|     | (Memory Cluster) |
              +--------+---------+     +--------+---------+
                       |                        ^
                       |   2. 跨节点高速传输    |
                       +====== KV Cache =======+
                                (RoCE RDMA / NIXL)
```

### 2. 请求全生命周期工作流

1. **请求接入**：全局 Router 接收到 Prompt 请求后，评估当前集群各节点的负载与前缀缓存命中情况。
2. **Prefill 算子执行**：请求被发送至 **Prefill 节点池**。Prefill 节点并行计算输入 Prompt，生成最初的 KV Cache。
3. **KV Cache 跨节点传输**：Prefill 节点通过高速网络（如 PCIe / RoCE RDMA / NIXL 库）将生成的 KV Cache 实时发送给指定的 **Decode 节点**。
4. **Decode 接管与流式返回**：Decode 节点收到 KV Cache 后，接管自回归计算，逐 Token 生成输出并通过 SSE/WebSocket 流式返回给用户，直到生成 EOS 标记。

---

## 三、 关键技术突破与核心挑战

### 1. KV Cache 跨节点传输优化 (Network Bandwidth & RDMA)

由于 Prefill 节点计算出的 KV Cache 必须传输给 Decode 节点，**网络传输带宽与延时**成为 PD 分离架构的关键生命线：

- **RDMA (Remote Direct Memory Access)**：绕过 CPU 内核协议栈，直接在两台 GPU 的 HBM / DRAM 之间进行数据拷贝，传输延迟控制在微秒级。
- **Chunked & Streaming Transfer（块流式传输）**：Prefill 节点在计算 Transformer 每一层或每个 Block 的 KV Cache 时，不必等待整批全部计算完，而是边计算边通过 RDMA 异步传输给 Decode 节点，实现**计算与网络传输的高度重叠 (Overlapping)**。

### 2. 分层存储与溢出机制 (Tiered Storage - 以 Mooncake 为代表)

月之暗面 (Moonshot AI) 开源的 **Mooncake** 系统提出了以 KV Cache 为中心的分层架构：

```text
HBM (GPU 显存)  <--->  DRAM (CPU 内存)  <--->  NVMe SSD (本地/分布式固态)
  [热 KV 缓存]            [温 KV 缓存]            [冷 KV 缓存]
```
- **HBM (GPU)**：存放当前正在进行 Decode 迭代的热页。
- **DRAM (CPU)**：当显存不够或节点切换时，冷页溢出至 CPU DRAM。
- **SSD**：针对极长上下文（100K+ Tokens），将历史文档的 KV 页持久化到 SSD，实现秒级加载与复用。

### 3. KV Cache 感知负载均衡 (KV-Aware Routing - 以 NVIDIA Dynamo 为代表)

全局路由器不仅根据 CPU/GPU 利用率调度，还基于全局前缀哈希树（Prefix Hash Tree）进行路由：
- **前缀复用最高优先**：若请求带有公共 System Prompt 或多轮对话历史，Router 会优先将其派发至已经缓存了该 Prompt KV 页的 Prefill/Decode 节点，避免重复计算。

---

## 四、 工业级代表系统对比

### 1. DistServe (ATC '24)
- **贡献**：首个系统化提出 Prefill 与 Decode 阶段解耦的推理系统。
- **成就**：在满足相同 SLO（首字延迟与字间延迟）约束下，相比传统共置系统，吞吐量提升了 **4.48 倍**。

### 2. Mooncake (Moonshot AI)
- **贡献**：月之暗面 Kimi 大模型生产环境的真实解耦推理架构。
- **成就**：以 KV Cache 为核心的分级解耦系统，长上下文场景下吞吐量最高提升 **525%**。

### 3. vLLM & SGLang 分布式 PD 解耦支持
- **贡献**：目前开源社区主流引擎在 2025/2026 年演进的核心方向。
- **成就**：支持通过 NIXL/Ray 通信库在多节点之间灵活组合视觉编码器解耦（Encoder Disaggregation）与通用 Decode 池。

---

## 五、 传统共置架构 vs PD 分离架构全方位对比

| 评估维度 | 传统共置架构 (Colocation) | PD 分离架构 (PD Disaggregation) |
| :--- | :--- | :--- |
| **集群资源部署** | 所有 GPU 节点混合运行 Prefill 与 Decode | 划分独立的 Prefill 节点池与 Decode 节点池 |
| **首字延迟 (TTFT)** | 受 Decode 竞争影响，易飙升 | 算力独占，TTFT 极低且可控 |
| **字间延迟 (TTPT)** | 易受突发 Prefill 请求打断，出现严重抖动 | 无 Prefill 打扰，TTPT 极其平稳 |
| **硬件配比灵活性** | 无法按算力/显存瓶颈单独扩容 | 可独立针对 Prefill（算力卡）与 Decode（显存卡）按需扩缩容 |
| **网络要求** | 单节点内 GPU 通信（NVLink） | 需要高带宽跨节点网络（RoCE RDMA / InfiniBand） |
| **系统复杂度** | 较低 | 较高（需全局调度器、网络传输库、状态恢复） |
| **集群吞吐量** | 基线 (1x) | **提升 2x ~ 5x** |

---

## 六、 总结与落地建议

1. **小规模/单机部署**：继续选用标准的 **vLLM / SGLang (PagedAttention)** 连续批处理共置模式，降低运维复杂度。
2. **大规模生产集群 (数据中心级)**：对于具备 RoCE RDMA 高速网络的基础设施，强烈建议升级至 **PD 分离架构**，将计算密集型与内存密集型算力解耦，大幅提升集群整体吞吐量与成本效益。

---
*原文参考：[今日头条 - 大模型推理服务：Prefill-Decode (PD) 分离架构](https://www.toutiao.com/article/7624403746868576819/?wid=1785914307657)*
