# 12. Tiny-LLM 附录：性能测试账本与优化归因分析（Performance Evidence Ledger）

> **导读**：本章节精读并完整翻译了 Alex Chi（skyzh）与 Connor Zhang 编写的高致密系统课程 **Tiny-LLM** 的核心附录 *Appendix: Performance Evidence Ledger*。
> 
> 该附录展示了使用 Apple Silicon MLX 框架与 Metal 算子（Metal Kernels）从零构建与优化 LLM Serving 推理引擎的全过程基准测试账本。它通过极度严谨的控制变量实验、算子耗时归因（Operator Attribution）与显存/带宽算力账本，解答了**“为什么课程按此顺序编排算子优化路线”**这一核心工程问题。
> 
> * **原文链接**：[Tiny-LLM Appendix: Performance Evidence Ledger](https://skyzh.github.io/tiny-llm/appendix-performance.html)
> * **开源代码参考**：[skyzh/tiny-llm (GitHub)](https://github.com/skyzh/tiny-llm)

---

> ⚠️ **状态说明：实验性单机测试数据**
> 在将任何正确性、集成度或性能结果视为普适性结论前，请先参阅 Week 2 验证矩阵。本附录记录了确定课程学习顺序的测量数据。需要注意的是，**各项性能数字并非可直接叠加的承诺**：当一个瓶颈被收缩后，其他算子占模型总运行时间的比例就会随之上升。

---

## 1. 基准测试方法（Benchmark Method）

性能递进测试器（Progression Runner）会在全新的进程中启动每一个 Checkpoint，交替它们的运行顺序，执行完整的请求预热（Warmup），在定时器内部同步 MLX 的惰性求值（Lazy Evaluation）任务，并报告中位数：

```bash
pdm run bench-week2-progression --offline --repeats 4 --cooldown-seconds 1 \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-128.json

pdm run bench-serving-progression --offline --repeats 4 \
  --model qwen3-4b --num-seqs 16 --batch-size 4 \
  --min-input-len 128 --max-input-len 1024 \
  --min-output-len 32 --max-output-len 128 \
  --prefill-step 128 --warmup 1 --cooldown-seconds 1 \
  --json-output benchmark_results/m4-pro-qwen3-4b-week3-serving-mlx-0.32.0.json
```

### 1.1 参数与测试规范
* `--prefill-logits last` 模拟文本生成服务负载（Generation-Serving Workload）：参考实现与 MLX 均只将 Prompt 的最后一行投影为词表 Logits。若用于 Prompt 评分请使用 `--prefill-logits all`，**严禁混用这两种模式进行对比**。
* **Decode 吞吐量**排除了生成的第一个 Token，因为首个 Token 是由 Prefill 阶段产生的。
* MLX 官方公布的 `mlx_lm.benchmark` 表格默认采用 **2,048 Token Prompt + 128 生成 Token**（即 2K/128），这适合作为静态库对比基准，而非 Paging 验收测试或长上下文证明。推荐采用上下文扫描（Context Sweep）：

| 序列长度 (Point) | 评估目的 (Purpose) |
| :--- | :--- |
| **128** | 固定的 Week 2 验收点以及短文本交互请求 |
| **2,048** | 标准 MLX 风格的静态压力对比点 |
| **8,192** | 长上下文注意力与 KV Cache 显存压力测试点 |
| **16,384** | 8K 路径稳定后的极限压力测试点 |

> 💡 `llama-bench` 默认采用 Prompt 512 + Decode 128，这再次提醒我们：基准测试长度是约定俗成的，而非通用工作负载。发布数据时**必须附带确切的 Prompt 与 Output 长度**。

### 1.2 测试硬件环境
测试机器为 **Apple M4 Pro（20 核 GPU，64 GB 统一内存）**：
* **Week 2 静态测试行**：2 次完整预热，4 个独立新进程平衡运行的中位数。
* **Week 3 连续 Serving 行**：1 次预热，4 个独立新进程平衡运行的中位数。

---

## 2. Week 2 Checkpoint 保留决策账本（Retention Ledger）

天花乱坠的解释不能作为优化项保留在课程中的证据。在决定保留一个 Checkpoint 之前，必须回答 6 个核心问题：**不变量是什么？为什么更快？在何处获胜？在何处失效？回退方案是什么？基准测试陷阱是什么？**

| 优化 Checkpoint | 必需不变量 | 性能假设 | 保留区间与失效形状 | 回退/对照组 (Fallback/Control) | 主要基准测试陷阱 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense KV Cache** | 调用方 offset 等于每层 Cache 长度；K/V 在序列轴追加 | 复用前缀投影 K/V，避免对全量前缀重复计算 | 随着前缀增长在 Decode 阶段获胜；但反复 `concat` 仍产生 $O(S^2)$ 字节拷贝 | Week 1 全前缀模型作为语义对照；Week 3 Paging 替代增长拷贝 | 用带 Cache 的 MLX 对比不带 Cache 的课程模型（测试的是不同算法） |
| **Packed Quantized Matvec** | W4, GroupSize 128, BF16 参数, 连续打包布局与转置约定 | 读取一次打包权重，在 SIMD Lane 间共享解包与 Scale 计算 | 在 $M \le 8$ 时保留；多行 Prefill 暴露复用率低的问题，引出 Day 6 | Python `mlx.core` 规范为正确性基准；原生 W4 Metal 为对照组 | 惰性求值或对已实例化权重计时会掩盖权重传输开销 |
| **RMSNorm** | BF16 输入输出，平方和在 FP32 中累加 | 将 Reduce 规约、归一化与权重乘法融合为单个 Dispatch | 在 Qwen 隐藏维度保留；未知维度需要重新测量 | Python `mlx.core` RMSNorm 与 Day 3 节点保持可切换 | 孤立累加微秒数，假定 Checkpoint 收益是彼此独立的 |
| **RoPE** | Batch 每行一个有效 offset；偶数旋转维度；保留尾部数值 | 融合角度生成与配对旋转，无需中间图结构 | 在 Qwen Decode 行保留；头数与旋转维度变更需重新测量 | Python `mlx.core` RoPE 与纯 RMSNorm 节点保持可切换 | 对比带 Cache/预计算角度路径与实时角度生成路径 |
| **SwiGLU** | Gate 与 Up 张量具有相同的 Shape 和 Dtype | 将 SiLU 激活与 Gate/Up 点乘融合为单个 Elementwise 调度 | 在 Qwen MLP 形状下保留；微型张量与其他 Dtype 不作性能承诺 | Python `mlx.core` SiLU 点乘与 RoPE 节点保持可切换 | 接受了单算子获胜，但全模型未获得重复收益 |
| **Decode Attention** | $H_q \% H_{kv} == 0$, $D \le 256$, FP32 Online-Softmax 状态 | 遍历 K/V 时避开 Score/Prob 张量并融合 Softmax | $L \le 2, S \le 256$ 无显式 Mask；Context 扫描在 256 内 6/6 全胜 | Python `mlx.core` 分组注意力处理更长 Query、更长 Context | 固定执行顺序、GPU 性能状态漂移、推演超出 256 |
| **SIMD-Matrix Prefill** | W4/Group-128 布局, BF16 存储, FP32 Tile 累加 | 在 Prompt 行之间复用 Activation 与反量化权重 Tile | $M > 8$ 时的必选路径；新模型 Shape 需要正确性与计时扫描 | Python `mlx.core` matmul 为正确性基准；Day 3 matvec 为短行调度 | 将全 Logit 课程 Prefill 与单 Logit MLX Serving 对比 |
| **Split-K Prefill** | 切分对齐量化 Group；局部平面互斥；最终归一化为 FP32 | 仅在普通结果网格未填满时增加独立 Threadgroup | 帮助短小狭窄的 Qwen 投影，128-Token 处中立，网格占满时亏损 | `split_k <= 1` 精确调度回 Day 6 未切分 Kernel | 剖析独立层会掩盖在依赖顺序模型中出现的占空率不足 |

---

## 3. Week 4 长上下文预算与极限推演（Long-Context Budget）

在 Qwen3-4B 模型中，单个 Token 的 BF16 K/V 状态显存占用为：
$$\text{36 层} \times 2 (\text{K and V}) \times 8 \text{ KV Heads} \times 128 \text{ Values} \times 2 \text{ Bytes} = 147,456 \text{ Bytes} = 144 \text{ KiB / Token}$$

### 3.1 显存与模型上限约束
* **模型训练与配置限制**：配置声明 `max_position_embeddings = 65,536`，但未配置 `rope_scaling`。官方文档指出 Qwen3 训练覆盖至 32,768 Tokens。因此未经修改的课程模型硬性验证上限为 **32,768 Tokens**。
* **M4 Pro (64 GB) 显存容量推算**：
  * GPU 推荐工作集：`5184 GiB`；量化模型占用：`1.99 GiB`；预留 Activation/Slack：`8 GiB`。
  * 显存理论容纳上限：
    $$\text{floor}\left(\frac{51.84 \text{ GiB} - 1.99 \text{ GiB} - 8 \text{ GiB}}{144 \text{ KiB}}\right) = 304,738 \text{ Tokens}$$
* **最终课程硬性预算**：取三者最小值：
  $$\min(32,768 \text{ Trained}, 65,536 \text{ Configured}, 304,738 \text{ Memory}) = \mathbf{32,768 \text{ Tokens}}$$

### 3.2 300K 上下文下的性能衰退规律
FlashAttention 消除了 $O(N^2)$ 的 Score 矩阵显存分配，但**并没有消除计算工作量**。Full-Attention Prefill 对上下文长度仍保持二次方计算量；Single-Token Decode 每一步必须读取线性增长的历史 KV。

下图展示了在 M4 Pro 上使用 MLX 0.32.0 对单条 Qwen3-4B BF16 Decode Query 进行的算子扫描：

| 上下文长度 (Context) | 全模型 BF16 KV 显存 | MLX SDPA 单层耗时 | 注意力理论 Decode 吞吐上限 |
| :--- | :--- | :--- | :--- |
| **2,048** | 0.28 GiB | 0.14 ms | 195.33 tok/s |
| **8,192** | 1.12 GiB | 0.29 ms | 96.72 tok/s |
| **32,768** | 4.50 GiB | 0.92 ms | 30.28 tok/s |
| **65,536** | 9.00 GiB | 1.73 ms | 16.08 tok/s |
| **131,072** | 18.00 GiB | 3.65 ms | 7.61 tok/s |
| **300,000** | 41.20 GiB | 9.49 ms | 2.93 tok/s |

---

## 4. Week 2 逐日（Day-by-Day）性能演进与算子归因

### 4.1 验收标准与逐日基准表格
* **固定验收 Shape**：Qwen3-4B, 128 Prompt Tokens, 128 Timed Decode Steps, Last-Row Logits, 2 Warmups, 4-Process Median。

| 章节 (Chapter) | 累积 Checkpoint | Prefill tok/s | Decode tok/s | Output tok/s | 前序归因选定的优化项 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Day 1** | Dense Request KV Cache | 730.43 | 24.63 | 24.01 | 停止全前缀 Decode 重复计算 |
| **Day 2** | Benchmark Baseline | 730.43 | 24.63 | 24.01 | 测量 Dense 投影权重传输 |
| **Day 3** | Quantized Matvec | 105.00 | **58.71** | 37.95 | 保持打包权重并引入 x4 Decode Kernel |
| **Day 4** | Fused Model Kernels | 105.97 | **75.21** | 44.33 | 消除刚暴露出来的 Pointwise 图发射 |
| **Day 5** | Bounded Decode Attention | 105.99 | **75.75** | 44.50 | 限制在 $S \le 256$ 内的 Online Softmax |
| **Day 6** | SIMD-Matrix Prefill | **797.45** | 75.12 | 69.17 | 修复 Day 3 暴露的量化矩阵 Prefill 路径 |
| **Day 7** | Split-K Prefill | 792.55 | 75.41 | 69.37 | 仅对低占用短投影充填 GPU |
| **Baseline** | MLX 0.32.0 | 830.49 | 89.37 | 81.30 | 外部参照分母 |

---

### 4.2 算子归因细节与优化推导逻辑

#### Day 2 归因：为什么先做 Quantized Matvec？
* 测量显示：在带 Cache 的 Decode 阶段，**Dense 投影算子占到了归因总时间的 81.5%（33.66 ms）**，而 Pointwise 算子占 15.6%，Attention 仅占 2.1%。
* **决策**：首要瓶颈是权重访存带宽！因此 Day 3 优先引入量化 Matvec Kernel。

#### Day 3 效果与微观对比（M=1 投影耗时）
引入打包 W4A16 Matvec 后，Decode 速度从 24.63 飙升至 **58.71 tok/s (+138.4%)**：

| Qwen3-4B 投影类型 (M=1) | 原生 Metal Kernel | 打包 Matvec (Day 3) | MLX 官方实现 |
| :--- | :--- | :--- | :--- |
| **Q 投影** | 750.3 us | **187.6 us** | 183.4 us |
| **K 投影** | 239.5 us | **145.1 us** | 147.8 us |
| **V 投影** | 244.8 us | **147.0 us** | 138.9 us |
| **O 投影** | 590.3 us | **163.7 us** | 160.2 us |
| **MLP Gate / Up / Down** | ~900-1200 us | **~182-188 us** | ~177-182 us |
| **Vocab Head** | 11,086.1 us | **1,030.2 us** | 1,029.3 us |

#### Day 4 归因：融合逐点算子 (Fused Pointwise Kernels)
* 当 Day 3 解决了权重带宽后，RMSNorm、RoPE 和 SwiGLU 激活函数的多次 Kernel Launch 散开开销上升至 **35.8%**。
* **优化结果**：
  * Fast RMSNorm：Decode 提升至 65.94 tok/s
  * Fast RoPE：Decode 提升至 71.16 tok/s
  * Fused SwiGLU：Decode 提升至 75.21 tok/s

#### Day 6 & Day 7 归因：Prefill 矩阵协同加载与 Split-K
* Day 3 虽然加速了 Decode，但导致 128-Token Prefill 从 730 tok/s 暴跌至 105 tok/s（因为矩阵 Prefill 走了低效路径）。
* **Day 6 SIMD-Matrix**：引入 $32 \times 32 \times 32$ 量化 Matmul 协同加载，Prefill 一举拉升至 **797.45 tok/s**！
* **Day 7 Split-K**：在 $M=32$ 等短 Prefill 场景下，由于线程组无法填满 GPU Compute Units，引入 Split-K 将 short-prefill 吞吐量再提升 **+11.9%**。

---

## 5. Week 3 PagedAttention 连续 Serving 性能表现

在连续 Serving 负载测试（16 个序列，Batch Size 4，输入 128-1024，输出 32-128）下，展示了从静态连续重建到 PagedAttention 的演进：

### 5.1 整体 Serving 吞吐与显存对比

| 存储与注意力路径 | Prefill tok/s | Output tok/s | Decode tok/s | 请求吞吐 (Req/s) | 峰值 KV 显存 | 避免的无谓 KV 拷贝量 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense 重建 (Day 2)** | 718.30 | 32.54 | 50.42 | 0.433 | 1,096 MiB | 209,532 MiB |
| **Paged + Gather (Day 3)** | 730.69 | 38.44 | 65.88 | 0.512 | — | 103,445 MiB |
| **Direct Paged Attention (Day 5)** | **679.56** | **41.88** | **82.11** | **0.558** | **576 MiB** | **504 MiB** |

### 5.2 核心结论
1. **直接 PagedAttention 相比 Dense Serving**：
   * **请求吞吐与 Output 吞吐提升 +28.7%**；
   * **Decode 聚合吞吐提升 +62.8%**（从 50.42 升至 82.11 tok/s）；
   * **峰值 KV 显存降低 47.4%**（从 1096 MiB 降至 576 MiB）；
   * **避免了 99.8% 的逻辑显存拷贝**（从 209 GB 剧降至 0.5 GB）！
2. **延迟控制**：Decode Step 中位数延迟从 58.95 ms 降低并稳定至 **38.27 ms**（P95 仅 39.83 ms）。

---

## 6. 全局优化路线图汇总（Optimization Map）

```
[测量瓶颈 1: 全前缀 Decode 重复计算] ────> [Day 1: Dense Request KV Cache]
                                                     │
[测量瓶颈 2: 矩阵投影权重带宽限制] ──────> [Day 3: Packed W4A16 SIMD Matvec]
                                                     │
[测量瓶颈 3: 频繁小 Kernel 启动开销] ──────> [Day 4: Fast RMSNorm / RoPE / Fused SwiGLU]
                                                     │
[测量瓶颈 4: 增长的短上下文 Attention] ────> [Day 5: Online-Softmax Decode Kernel]
                                                     │
[测量瓶颈 5: 标量/跨步 Prefill 投影加载] ───> [Day 6: 32x32x32 Cooperative Quantized Matmul]
                                                     │
[测量瓶颈 6: 未填满的 Short-Prefill 网格] ───> [Day 7: Measured Split-K Dispatch]
                                                     │
[测量瓶颈 7: 动态 Batch 连续重构内存拷贝] ──> [Week 3: Paged KV Cache & Direct Paged Attention]
```
