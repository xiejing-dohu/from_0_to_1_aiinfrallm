# 4_vLLM高吞吐大模型推理系统架构与核心技术深度解析

> **源链接**：[Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm) （作者：Aleksa Gordić，发表时间：2025-08-29）  
> **摘要**：本文全面拆解了现代高吞吐大语言模型（LLM）推理系统 vLLM 的底层架构与核心机制。从基础的 PagedAttention、连续批处理（Continuous Batching）与调度策略，到 Chunked Prefill、前缀缓存（Prefix Caching）、语法约束解码（Guided Decoding）、投机解码（Speculative Decoding）与 PD 分离架构，再到多 GPU/多节点分布式动态 Serving 扩展，最后探讨了推理系统的性能指标与自动调优。

---

## 概述与导读

vLLM 是当前大模型推理与 Serving 领域的基石系统之一。本文遵循“倒金字塔”结构，由浅入深逐步解构 vLLM 的系统组成与高级特性：

1. **LLM Engine & Engine Core**：vLLM 的基础组件（调度器、PagedAttention、连续批处理等）。
2. **高级特性（Advanced Features）**：Chunked Prefill、前缀缓存（Prefix Caching）、语法约束解码（Guided Decoding）、投机解码（Speculative Decoding）、PD 分离。
3. **规模化扩展（Scaling Up）**：从单 GPU 到多 GPU / 张量并行（TP）与流水线并行（PP）执行。
4. **服务层（Serving Layer）**：分布式与高并发 Web 服务脚手架（FastAPI + ZMQ + Ray/DP）。
5. **基准测试与自动调优（Benchmarks & Auto-Tuning）**：吞吐量与延迟的权衡与测量。

> **背景说明**：分析基于 vLLM [Commit 42172ad](https://github.com/vllm-project/vllm/tree/42172ad)（2025年8月），重点聚焦于 **V1 引擎架构**。

---

## 一、 LLM Engine & Engine Core 核心架构

LLM Engine 是 vLLM 的核心构建块。在离线推理模式下，LLM Engine 已经具备了极高的计算吞吐能力。

### 1.1 离线推理基础示例

以下是一个最基础的单进程、单 GPU 离线推理示例：

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
    main()
```

该配置处于：
- **离线模式**（无 Web/分布式服务脚手架）
- **同步执行**（在单个阻塞进程中运行）
- **单 GPU**（无 DP/TP/PP/EP Parallelism）

### 1.2 LLM Engine 构造过程

LLM Engine 主要由以下四大核心模块组成：

```text
+-----------------------------------------------------------------------+
|                               LLM Engine                              |
|                                                                       |
|  +---------------+  +---------------+  +---------------+  +----------+|
|  |  vLLM Config  |  |   Processor   |  |EngineCoreClient|  | Output   ||
|  | (模型/缓存/并行)|  |(验证/分词/转换) |  | (Inproc/DPLB) |  |Processor ||
|  +---------------+  +---------------+  +---------------+  +----------+|
|                                                |                      |
|                                                v                      |
|                                 +----------------------------+        |
|                                 |        Engine Core         |        |
|                                 |                            |        |
|                                 |  +----------------------+  |        |
|                                 |  |    Model Executor    |  |        |
|                                 |  | (UniProc/MultiProc)  |  |        |
|                                 |  +----------------------+  |        |
|                                 |  +----------------------+  |        |
|                                 |  |Structured Output Mgr |  |        |
|                                 |  +----------------------+  |        |
|                                 |  +----------------------+  |        |
|                                 |  |      Scheduler       |  |        |
|                                 |  | - FCFS / Priority    |  |        |
|                                 |  | - Waiting / Running  |  |        |
|                                 |  | - KV Cache Manager   |  |        |
|                                 |  +----------------------+  |        |
|                                 +----------------------------+        |
+-----------------------------------------------------------------------+
```

在构造 `Model Executor` 时，系统会创建 `Worker` 对象并执行三个核心初始化步骤：

1. **设备初始化（Init Device）**：
   - 绑定 CUDA 设备，校验数据类型（如 bf16）。
   - 检查并分配显存比例（如 `gpu_memory_utilization = 0.8`）。
   - 初始化 `model_runner`（包含 Sampler、KV Cache 与前向传播 Buffer）与 CPU 侧 `InputBatch`。
2. **加载模型（Load Model）**：
   - 实例化模型架构，加载权重，切换至 `model.eval()` 模式（可选执行 `torch.compile`）。
3. **初始化 KV Cache**：
   - 获取每层的 KV Cache 规格（如标准 Transformer 的 `FullAttentionSpec`）。
   - **单 Block 显存计算标准**：`2 (Key/Value) * block_size(16) * num_kv_heads * head_size * dtype_bytes(2)`。
   - 运行 Profiling 预热前向传播，计算显存中可容纳的 KV Block 数量并分配显存。
   - 预热并捕获不同 Batch Size 下的 **CUDA Graphs**，规避运行时 Kernel Launch 开销。

### 1.3 `generate()` 执行流程与连续批处理（Continuous Batching）

请求进入引擎后被转换为 `Request` 对象放入 Scheduler 的 `waiting` 队列，状态标记为 `WAITING`。引擎循环调用 `step()` 方法，每个 step 分为三个阶段：

```text
                 +-----------------------------------+
                 | 1. Schedule (调度)                |
                 | - 选择 decode / prefill 请求      |
                 | - 动态分配 KV Cache Block         |
                 +-----------------+-----------------+
                                   |
                                   v
                 +-----------------------------------+
                 | 2. Forward Pass (前向传播)        |
                 | - 打平 Batch 拼接为 Super-Sequence |
                 | - 执行 PagedAttention Kernel      |
                 | - Logits 采样生成新 Token         |
                 +-----------------+-----------------+
                                   |
                                   v
                 +-----------------------------------+
                 | 3. Postprocess (后处理)           |
                 | - 拼接 Token，反词元化 (Detokenize)|
                 | - 检查停止条件，释放 KV Block     |
                 +-----------------------------------+
```

- **停止条件**：达到 `max_tokens` / `max_model_length`；生成 EOS Token；匹配 `stop_token_ids`；命中 `stop` 字符串。

---

## 二、 调度器（Scheduler）与 PagedAttention 机制

### 2.1 Prefill 与 Decode 混合调度

Workloads 主要分为两类：
- **Prefill（首字阶段）**：对 Prompt 所有 Token 并行计算，属于 **计算密集型（Compute-bound）**。
- **Decode（生成阶段）**：单 Token 增量计算，需加载全量模型权重与历史 KV Cache，属于 **内存带宽密集型（Memory-bandwidth-bound）**。

vLLM V1 调度器优先保障处于 `running` 队列的 Decode 请求：
1. 优先为 Decode 请求调用 `kv_cache_manager.allocate_slots` 分配新 Slot。
2. 剩余 Token Budget 分给 `waiting` 队列的 Prefill 请求。
3. 若显存不足，触发抢占（Eviction/Recompute），释放低优先级请求的 KV Block 回收至 `free_block_queue`。

### 2.2 PagedAttention 映射逻辑

PagedAttention 模仿操作系统的虚拟内存页管理：
- 将 KV Cache 拆分为固定大小的 Block（默认 16 个 Token）。
- 通过 `req_to_blocks` 映射表将请求的逻辑 Token 序列映射到非连续的物理显存 Block 中。
- 前向传播时将 Batch 中的所有序列打平拼接成一根长的 "Super sequence"，配合 `slot_mapping` 与 Position ID，无需 Right-Padding 即可高效并行计算。

---

## 三、 高级系统特性（Advanced Features）

### 3.1 Chunked Prefill（分块预填）

当 Prompt 极长时，全量 Prefill 会长时间占用 GPU 算力，导致处于 Decode 阶段的其他请求严重卡顿（ITL 飙升）。

- **原理**：将长 Prompt 按照固定阈值（`long_prefill_token_threshold`）切分成多个块（Chunks）。
- **效果**：在多个 engine step 中分批完成 Prefill，允许 Decode 请求交错执行，大幅平滑了系统的字间延迟（ITL）。

### 3.2 Prefix Caching（前缀缓存）

对于存在相同系统提示词（System Prompt）或多轮对话场景，Prefix Caching 避免重复计算 KV Cache：

```text
[ 请求 1 ]:  |--- 共享系统前缀 (System Prompt) ---| + |--- 提示词 A ---|
                                │
                        计算并哈希存入 Cache
                                │
                                v
[ 请求 2 ]:  |--- 共享系统前缀 (System Prompt) ---| + |--- 提示词 B ---|
                                │
                          直接命中并复用 Block
```

1. **BlockHash 计算**：将 Prompt 按 16 Token （BlockSize）分块，计算 Hash（Hash 包含上一个 Block 的 Hash、当前 Tokens 及元数据如 Cache Salt）。
2. **Cache Hit**：检查 `cached_block_hash_to_block`。若命中，直接复用已有的物理 KV Block，引用计数 +1，跳过这部分 Token 的 Prefill 计算。

### 3.3 Guided Decoding（基于 FSM 的语法约束解码）

在结构化输出（如 JSON、特定 Regex）场景下，vLLM 集成了语法约束引擎（如 XGrammar）：
- **初始化**：构造引擎时初始化 `Structured Output Manager`。
- **运行机制**：在每一个 Decode Step，有限状态自动机（FSM）根据当前生成状态遮蔽（Mask）掉不符合语法的 Token Logits。
- **概率重采样**：将不允许出现的 Token 对应概率设置为 $-\infty$，确保采样出来的下一个 Token 100% 严格遵循指定的语法树或正则表达式（如 Chomsky Type-2 上下文无关文法）。

### 3.4 Speculative Decoding（投机解码）

投机解码通过小模型（Draft Model / N-gram）与大模型（Target Model）结合加速验证：

```text
1. Draft 阶段:   Draft Model 快速连续生成 K 个 Candidate Tokens
                 [ Token 1, Token 2, Token 3, ..., Token K ]
                                 │
                                 v
2. Verify 阶段:  Target Model 单次前向传播并行验证这 K 个 Tokens
                                 │
                                 v
3. Accept/Reject: Rejection Sampler 接受前 M (M<=K) 个符合分布的 Token
                  并修正第一个被拒绝的 Token
```

- **逻辑步骤**：
  1. **初始化**：模型执行器构造时创建 `drafter`（Draft Model，如 `NgramProposer`）与基于 Triton 实现的 `rejection_sampler`。
  2. **生成候选**：完成大模型 Prefill 后，调用 `propose_draft_token_ids(k)` 提出 $K$ 个 Draft Tokens 存入 `request.spec_token_ids`。
  3. **预留显存**：下一个 Step 中 Scheduler 将 $K$ 加入预留 Tokens，显存分配器预留对应数量的 KV Block。
  4. **并行验证与拒绝采样**：Target Model 执行一次包含上下文和 Candidate Tokens 的前向传播，通过 `rejection_sampler` 从左到右依次接受/拒绝，修正首个不合规 Token。
- **效果**：将多次串行 Decode 前向传播转化为一次并行验证前向传播，显著降低内存带宽瓶颈带来的延迟。

### 3.5 Disaggregated P/D（PD 分离架构）

由于 Prefill（算力受限，Compute-bound）与 Decode（带宽受限，Memory-bandwidth-bound）对硬件要求截然不同，PD 分离将两者部署在不同的计算节点上：

```text
                   +------------------------+
                   |   Prefill 节点 (GPU 0) |
                   |  (专门负责首字并行计算)  |
                   +-----------+------------+
                               |
                   KV Cache 传输 (SharedStorage / NIXL/RDMA)
                               |
                               v
                   +------------------------+
                   |   Decode 节点 (GPU 1)  |
                   |  (专门负责逐字生成解码)  |
                   +------------------------+
```

- **生命周期与上下文交互**：
  1. **实例化**：系统在 Worker 初始化与 Scheduler 构造时分别实例化 role="worker" 和 role="scheduler" 的 Connector（如 `SharedStorageConnector` 或 `LMCache`）。
  2. **外部 Cache 检索**：Scheduler 调度 Prefill 请求时，通过 `connector.get_num_new_matched_tokens` 检索外部 KV 缓存服务器。
  3. **前向传播上下文管理**：进入前向传播前，上下文管理器执行 `kv_connector.start_load_kv`（Decode 实例从外部传输并注入物理 KV 页显存）；前向传播退出时执行 `kv_connector.wait_for_save`（Prefill 实例阻塞等待 KV 保存上传完成）。

---

## 四、 规模化扩展：MultiProcExecutor 与分布式 Serving

### 4.1 MultiProcExecutor（单节点/多节点张量并行）

当单个 GPU 显存无法容纳模型时，需要使用张量并行（Tensor Parallelism, TP）：

```text
                   +----------------------------------+
                   |      MultiProcExecutor (主进程)  |
                   +----------------+-----------------+
                                    |
                           rpc_broadcast_mq (共享内存)
                                    |
         +--------------------------+--------------------------+
         |                                                     |
         v                                                     v
+------------------+                                  +------------------+
| Worker Rank 0    |                                  | Worker Rank 7    |
| (Driver Worker)  |                                  | Worker Rank 7    |
| - CUDA Device 0  |                                  | - CUDA Device 7  |
| - Exec Forward   |                                  | - Exec Forward   |
+--------+---------+                                  +--------+---------+
         |                                                     |
         +--------------------> worker_response_mq <-----------+
```

1. **多进程握手与建立**：主进程启动 `MultiProcExecutor`，通过 `WorkerProc.make_worker_process` 为各个 Rank 派生进程。主进程与子进程建立 Pipe 传输 `worker_response_mq` 句柄，初始化共享内存消息队列 `rpc_broadcast_mq`。
2. **Worker 忙等待循环**：所有 Rank Worker 阻塞在 `rpc_broadcast_mq.dequeue`。当主进程放入前向传播指令时，各 Worker 触发执行 partitioned work，并通过 NCCL 进行通信同步。
3. **响应返回**：非 Driver Worker 执行计算，Driver Worker (Rank 0) 将 Logits 采样结果推入 `worker_response_mq.enqueue` 返回主进程。

### 4.2 分布式 Serving 架构拓扑与线程模型

在生产环境中，系统通常采用多节点多副本（Data Parallelism, DP）架构部署：

```text
                  +--------------------------------+
                  |    Client / HTTP Request       |
                  +---------------+----------------+
                                  |
                                  v
                  +--------------------------------+
                  |  FastAPI / Uvicorn API Server  |
                  |  (OpenAIServingCompletion)     |
                  +---------------+----------------+
                                  |
                        AsyncLLM (DPAsyncMPClient)
                                  |
            +---------------------+---------------------+
            |                                           |
            v                                           v
  +-------------------+                       +-------------------+
  |  DP Coordinator   |                       | Load Balancer     |
  | (心跳/伸缩/Wave)   |                       | (按 Queue 长度分发)|
  +---------+---------+                       +---------+---------+
            |                                           |
            +---------------------+---------------------+
                                  | ZMQ (Dealer Sockets)
                                  v
       +--------------------------+--------------------------+
       |                                                     |
       v                                                     v
+------------------------------+              +------------------------------+
| DPEngineCoreProc (Node 1)    |              | DPEngineCoreProc (Node 2)    |
| - Engine Core (TP=4)         |              | - Engine Core (TP=4)         |
| - Input / Main / Output Thread|              | - Input / Main / Output Thread|
+------------------------------+              +------------------------------+
```

#### 细节拆解与多线程工作流：

1. **Headless Engine 节点进程/线程模型**：
   - 每个 DP Replica 运行一个 `DPEngineCoreProc` 进程。
   - **Input Thread**：阻塞监听 ZMQ Socket。收到 API Server 路由的 Request 后反序列化，通过 `input_queue.put_nowait(...)` 非阻塞推入主线程队列。
   - **Main Thread**：阻塞等待 `input_queue.get()`，喂入 `EngineCore` 循环调用 `step()` 进行前向传播计算与采样，并将结果推入 `output_queue`。
   - **Output Thread**：阻塞等待 `output_queue.get()`，通过 ZMQ Socket 异步将 Token 结果发回 API Server 节点。
   - **DP 锁步与 Dummy Step**：在混合并行模式（如 MoE 模型的 EP/TP）下，DP 节点间若有副本空闲，会配合执行 Dummy Step，防止由于 NCCL 规约阻塞活跃副本。

2. **API Server 节点与 AsyncIO 任务协同**：
   - **FastAPI / Uvicorn**：暴露 OpenAI 兼容接口，解析 HTTP 请求。
   - **AsyncLLM & DPAsyncMPClient**：维持负载均衡算法，公式计算引擎得分 $Score = \text{len}(waiting) \times 4 + \text{len}(running)$，将请求发往低负载 Engine。
   - 内部维持三个并行 AsyncIO 异步任务：
     - `create_completion`：处理 HTTP 请求生命周期。
     - `process_outputs_socket` & `output_handler`：监听 Engine Output Socket 并组装流式响应。
     - `run_engine_stats_update_task`：与 DP Coordinator 通信轮询集群负载与弹性伸缩（`SCALE_ELASTIC_EP`）。

---

## 五、 基准测试与性能调优（Benchmarks & Auto-tuning）

### 5.1 推理核心性能指标

| 指标 | 全称 | 定义与说明 |
| :--- | :--- | :--- |
| **TTFT** | Time To First Token | 从提交请求到收到**第一个输出 Token** 的时间（反映 Prefill 性能）。 |
| **ITL** | Inter-Token Latency | 连续两个输出 Token 之间的生成间隔时间（反映 Decode 性能）。 |
| **TPOT** | Time Per Output Token | 单个请求所有输出 Token 的平均 ITL。 |
| **E2E Latency** | End-to-End Latency | 端到端总延迟：`TTFT + sum(ITL)`。 |
| **Throughput** | 吞吐量 | 单位时间内系统处理的总 Token 数或请求数（Tokens/s 或 RPS）。 |
| **Goodput** | 有效吞吐量 | 满足业务 SLO（如限制 Max TTFT < 200ms）的合格 Token 吞吐量。 |

### 5.2 Latency vs. Throughput 权衡关系与 Roofline 模型

```text
    性能 (FLOPs / Step)
           ^
  峰值算力  │                        /----------------------- (Compute-Bound 区域)
  P_peak   │                       /
           │                      /
           │                     / 
           │                    /  
           │                   /   
           │                  /    
           │                 /     
           │----------------/ (Memory-Bound 区域)
           +-------------------------------------------------->
                           B_sat (饱和 Batch Size)             Batch Size (B)
```

- **Batch Size $B \to 1$**：系统处于 Memory-bound 状态，ITL 最低，适合极低延迟交互场景。
- **Batch Size $B \to \infty$**：矩阵乘法效率显著提高，系统向 Compute-bound 转移。单位时间吞吐量达到峰值，但单请求的 ITL 会有所上升。

### 5.3 vLLM 性能测试命令示例

```bash
# 测试延迟 (Latency Benchmark)
vllm bench latency \
  --model <model-name> \
  --input-tokens 32 \
  --output-tokens 128 \
  --batch-size 8

# 测试吞吐量 (Throughput Benchmark)
vllm bench throughput \
  --model <model-name> \
  --dataset <path-to-sharegpt-json>

# 仿真真实 Web 流量服务测试 (Serve Benchmark)
vllm bench serve \
  --model <model-name> \
  --request-rate 10.0
```

---

## 总结

vLLM 通过 **PagedAttention** 解决了显存碎片的根本痛点，并结合 **连续批处理**、**Prefix Caching**、**Chunked Prefill**、**投机解码** 与 **PD 分离** 等一系列工程创新，构建了高吞吐、低延迟的现代大模型推理服务范式。通过 **MultiProcExecutor** 与 **DP 协调调度器**，vLLM 能够平滑扩展至多卡与多节点集群，满足高并发工业级生产需求。
