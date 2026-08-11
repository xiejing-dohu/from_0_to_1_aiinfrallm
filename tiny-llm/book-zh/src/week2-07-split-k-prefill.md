# 🚧 第 2 周第 7 天：Split-K Prefill

> **现状:实验。** 见
> [第2周核查矩阵](./week2-overview.md#verification-status) (单位:千美元)
> 不断测试、在当地测量和仍在审查中的内容。

第6天的合作载荷带来了长距离 prefill 近处 MLX它的后续扫描显示的是不同的
问题在短处 prefill: Qwen狭窄的 K/ V 预测发射不足
独立结果牌 GPU。今天,我们分担削减
直到电网足够大为止

本章不是一般的拆分K库. 它优化了模型的形状
实际运行 :

| 型号 | 减少 `N` | Q 输出 `K` | K/V 输出 `K` |
|---|---:|---:|---:|
| Quen3 - 4B | 2,560 | 4,096 | 1,024 |

## 为什么要分拆减幅?

为: `C = A @ W.T`第6天发布:

```plain
ceil(M / 32) * ceil(K / 32) threadgroups
```

Split-K 添加分区网格尺寸 :

```plain
partial[p, :, :] = A[:, N_start[p]:N_end[p]]
                    @ W[:, N_start[p]:N_end[p]].T
C = reduce(partial, partition axis)
```

这暴露了更独立的工作,但重新阅读了部分 `A`,分配一个
并发射还原内核。 只有在
原二维网格填充不足.

## 任务1:重置未填充网格

任务1不改变函数 。 现有基准
`quantized_matmul_simdgroup_w4a16_g128` 第6天 编辑 Split-K 前的内核
s.

从窄 K 投影开始 `M=32` 在更改调度之前。 这是
复制填充不足的第6天网格所需的最小基线; 任务 4
运行全部投影, Split- K 存在后全行扫描 :

```bash
pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
  --section prefill-projections --context 32 --prefill-projection k \
  --warmup 5 --iterations 30
```

记录同步第6天和 MLX 实施Split-K之前的延迟。 那个
狭长的K/V形状是最清晰的小网格箱. 输出宽度大或提示
长度可能已经有足够的逐列的瓦片,应该成为控制器。

## 任务2:每个分区重用第6天核心

执行 `quantized_matmul_simdgroup_splitk_w4a16_g128` 输入
`src/extensions/src/quantized_matmul.metal`,重用第6天的平板助手
后面 `quantized_matmul_simdgroup_w4a16_g128`.

添加 `group_id.z` 作为分区索引。 每个分区必须:

- 缩短长度相同;
- 开始和结束于128个值的量化-组边界;
- 重新使用经过验证的第6天装载机、减压器和32x32瓦;
- 写信给自己 `[M, K]` 飞机没有漫画。

将部分飞机存放在 BF16 保存临时小块并进行最后
合计 FP32 在输出之前。 这又加了一个 BF16 四舍五入
与未分割的边框比较 FP32 累积器,所以测试使用
BF16合适的宽容. 一个 FP32 暂时性是个有用的预言,但
它使部分阻塞流量翻倍.

## 任务 3: 从占用中选择分区

修改 `QuantizedMatmul::eval_gpu` 输入
`src/extensions/src/quantized_matmul.cpp` 选择分区数,并
发送 Split-K 内核 。 保留 `tiny_llm_ext::quantized_matmul` 联 合 国
Python 约束不变; 现有 `use_split_k` 参数带有此特性
累计检查站。

使用一个小的明确政策:

```plain
base_groups = ceil(M / 32) * ceil(K / 32)
split_k = min(16, floor(320 / base_groups), N / 128)
decrease split_k until N is divisible by split_k * 128
use Day 6 unchanged when split_k <= 1
```

对于 Quen3-4B 目标, 大约使用 320 线组
和16的上限作为明确的调制参数。 它们不是普遍的 GPU
属性。 与硬编码的即时长度截断不同,网格计算
一旦再出现一排瓷砖, 自然停止分割一个窄的投影,
并立即停止 已经宽的网格。

对于 Qune3-4B , 策略选择这些列表 :

| 预测 | 基地组 `M=32` | 选中的拆分时间 `M=32` | 选中的拆分时间 `M=128` |
|---|---:|---:|---:|
| Q, `2560 -> 4096` | 128 | 2 | 1 |
| K/V, `2560 -> 1024` | 32 | 10 | 2 |
| O, `4096 -> 2560` | 80 | 4 | 1 |
| MLP门/上, `2560 -> 9728` | 304 | 1 | 1 |
| MLP下来, `9728 -> 2560` | 80 | 4 | 1 |

一分为二意味着调度员使用第6天内核不变. 在那个
只有狭窄的K/V预测才符合条件,
双向分割;其他主要预测已经显示出足够的产出
瓷砖。 在2,048个令牌中,每个投影都使用未分裂的内核.

通过累积显示策略 `split-k` 检查站。 保留第6天
可选择, 因此基准总是有未分割的控制 。

## 任务4:减少和核实

执行 `quantized_matmul_splitk_reduce` 输入
`src/extensions/src/quantized_matmul.metal` 并完成相应的
减少发送量 `QuantizedMatmul::eval_gpu`不添加第二个公开
matmul 函数。

每个输出元素发射一个还原线. 校对全部分区值
FP32 一次投给模特儿 dtype测试 :

- Quen3 -4B的车 `2560 -> 1024` K/V预测;
- 部分32列输出瓦;
- 其基网已经达到320组并因此倒塌的形状
  完全到第6天为止。

```bash
pdm run build-ext
pdm run test --week 2 --day 7

for context in 16 32 64 128 2048; do
  for projection in q k v o gate up down; do
    pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
      --section prefill-projections --context "${context}" \
      --prefill-projection "${projection}" --include-split-k
  done
done
```

## 基准分析:完成第2周

比较第6天,第7天和 MLX 时间短 接受 时间长
Split-K只在未分割输出网格填充不足时才会有所帮助. 校验
那个单键 decode 没有改变,因为它仍然 发送到第3天
足够大 prefill 形状选择未分割的第 6 天
内核而不是支付部分存储和削减.

在填充不足的形状扫荡旁保持一个短的完整模型控制,然后
从第3天开始运行固定的第2周接收工作量。 那个
[业绩附录](./appendix-performance.md) 是一个
计量的硬件、依赖性版本、检查站表和最终数据 MLX 比率。

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-simd-matmul --variant week2-split-k --variant mlx \
  --model qwen3-4b --input-len 32 --output-len 33 --warmup 2 \
  --prefill-logits last

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-simd-matmul --variant week2-split-k --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-simd-matmul --variant week2-split-k --variant mlx \
  --model qwen3-4b --input-len 2048 --output-len 129 --warmup 2 \
  --prefill-logits last
```

重复128个收件形状和长时间的操作员比较
控制,例如2,048个令牌。 附上三个端对端比较和
每个预测 SIMD/Split-K/ MLX 每个交叉候选人的表格。 保留
仅在测量到的交叉面以下拆分 K : 它必须改善填充不足的情况
投影, 保存单键 decode回到第六天时
普通结果网格已被占用 。 记录累积和减少
运算分区政策和运算表旁的发送。 决赛
延伸目标接受率仍然必须达到80%
页:1 MLX 在两个阶段。

[参考检查站](./appendix-performance.md#day-7-split-k-only-below-the-crossover)
将短形状操作员增益与端到端结果对齐,并保留
中性接受和长控分开。 直接验证短处
形状执行累积和合并管道,而计算
策略名称分区和形状扫描防止其顶端
泄漏到被占用的控制。 第3周则改变基准本身:
请求周转和密集的 KV 重建,而不是另一个静态
成为被测量的瓶颈

周2环现已完成:

```plain
optimize matvec -> benchmark decode -> optimize model kernels -> benchmark decode
-> optimize attention -> benchmark prefill -> optimize cooperative matmul
-> measure tile occupancy -> optimize split-K -> benchmark the complete checkpoint
```

第3周继承了这些投影时间表. 单独评价
缓存写、 直接页面读取、 注意时间、 端到端的吞吐量; 它
第7天的投影收益,不计为利息。

{{#include copyright.md}}
