# 🚧 第 3 周第 5 天：分页 FlashAttention

>            本章正在审查中,可能有所改动。

在本章中,我们将对多键查询给予注意。
运算符通过 `block_table`,阶段
页面后置的芯片,并将其与 online softmax。简短查询
继续使用矢量 decode 从第4天起; 长 prefill 块使用
这里开发的平板表。

这是需要的一章。 FlashAttention 属于这里而不是第二周
因为服务模型真正的K/V源现在是页面池. 建设a
密集的内核首先会创建第二个关注路径,然后需要
学生们将围绕页面翻译重新学习其内存时间表.

## 先决条件

本章综合了四个先决条件:

- 第2周第5天引入了在线软件重现。
- 第6天第2周,介绍合作社建造的32×32瓦 BF16 8×8
  SIMD-马特里克斯碎片.
- 第3周第3天引入了物理页面和块表。
- 第3周,第4天 引入直接的页面浏览关注和 decode
  时间表。

无新模式 dtype 这里介绍。 保留第2周的精确合同
现场 `paged_attention` 边界。

## 为什么优化页面路径

常规关注表达式实现带有形状的分数矩阵
`L × S`。一个页面行走的执行可以避免收集 K/V 并且仍然可以
中间体太大了 页 次 FlashAttention 两者都:

1. 通过 `block_table` 解析每个 K/V 分块，而不是收集一个稠密的
   缓存;
2. 芯片上只保留一个查询瓦、一个K/V瓦和在线软max状态;
3. 在所有可见的页面被耗尽后,它会写出正常的输出。

算法仍然是精确的注意. 只有负载和减少的顺序
变化。

## 保留第4天接口

不要添加第二个模型显示操作员 。 继续打电话:

```python
paged_attention(
    query,
    key_pages,
    value_pages,
    block_table,
    context_lens,
    page_size,
    scale=scale,
    mask="causal",
)
```

将形状调度放入扩展名 :

| 查询形状 | 时间表 |
|---|---|
| `L <= 8` | 保留 Day 4 矢量页码内核 。 |
| `L > 8`, BF16, `D == 128` | 使用平面页 FlashAttention 内核. |

因此,已完成的第3周模式有1份网页访问合同和2份
具体工作量 GPU 时间表。

## 任务 1: 文字查询和页 K/V

开始 `paged_attention_mma_bf16_d128` 输入
`src/extensions/src/paged_attention.metal`保留
`paged_attention_decode` 和 `paged_attention_scalar_f32` 从第4天起,未变;
它们仍然是简洁和通用的控制。

使用 8 SIMD 组以覆盖64行查询块。 每个 SIMD 组拥有 8
查询行并代表矩阵操作符为8×8片段. 第32阶段逻辑
每个迭代的K/V位置.

对于砖块中的每个逻辑键行:

```plain
logical_position = tile_start + row
logical_page     = logical_position / page_size
slot             = logical_position % page_size
physical_page    = block_table[batch, logical_page]
address          = pages[physical_page, kv_head, slot, :]
```

打开瓦片时解析物理页面 。 矩阵乘数应
不知两个相邻的逻辑行是否来自相邻的物理页面。

那个 Qwen 路径使用128个token页面和一个32个token K/V 瓦。 对齐的瓦片是
因此,即使整个逻辑序列是
没有 通过一个合作区块加载器指定每个线条毗连元素
因此,相邻的车道 发行连接读物。 为瓦片保留通用装载器
跨越页面边界。  Metal 可使用内核 MLX低级钢铁
此负载原始的块装填器头, 而您的解决方案拥有此页面
翻译,瓦片表, online softmax原始的,和调度。 没错
非即时 MLX 注意 注意点

需要有尾矿。 查询块、 K/V 瓦、 最终页或上下文可能
部分内容完整,物理页码无需连续.

## 任务 2: 计算倾斜 Online Softmax

继续修改 `paged_attention_mma_bf16_d128` 输入
`src/extensions/src/paged_attention.metal`。此任务将填充平板
在线 softmax;它不添加另一个公共功能。

对于每个查询牌,保持一个最大运行,一个运行总和,以及一个
每行非正常输出累积器 。 每个K/V瓦:

1. 计算 `Q @ Kᵀ` 与第二周SIMD-矩阵碎片;
2. 适用规模和因果关系;
3. 将最大瓦片并入运行中的最大瓦片;
4. 调整先前的总和和产出累积器;
5. 计算当前分数的指数并更新运行中的总和;
6. 将瓦片概率乘以V并更新输出积分器.

在最后可见的瓦片之后,将每个输出行除以运行中的总和和和
使用模型格式存储 dtype.

将注意力比例乘以 `log2(e)` 一次并使用 `fast::exp2` (单位:千美元)
热瓦循环内部的在线 softmax缩放。 这是数学上的
等同自然指数,并避免重复基质转换。

乘以因数 `context_len - L`。在逻辑位置键 `s` 可见
以查询行 `l` 当:

```plain
s <= l + context_len - L
```

当第一个键超过最后一个可见键时跳过整个 K/V 瓦
查询块。 这既是正确规则,也是重要的因果预告
优化。

## 任务3:验证页面边界

完成长清水选择 `PagedAttention::eval_gpu` 输入
`src/extensions/src/paged_attention.cpp`测试
`paged_attention_mma_bf16_d128` 对抗第四天的内核 保留
`tiny_llm_ext::paged_attention` 页:1 Python `paged_attention` 签名
不变。

从第2周第3天开始使用 GPU 调试梯子 :

1. 将第4天的注意页与可读方程式相比较
   与 `mlx.core`;
2. 比较页 FlashAttention 以第四日的道路发誓,
3. 仅以此为基数。

所需固定装置包括:

- 一页内的内容;
- 跨越页数界限的瓦片;
- 非连续物理页码;
- `L = 65` 和长度不是多瓦的上下文;
- 因果关系 decode 页面后 prefill;
- GQA,其中多个查询头映射到一个K/V头;
- 产出 dtype 遗 物 BF16.

强制 `mx.eval` 在每个操作员之后立即进行汇编、发送和
报告处理失败的情况是负责任的。

```bash
pdm run test --week 3 --day 5
```

## 任务4:整合和措施

验证当前发送到 `Qwen3MultiHeadAttention.__call__` 页:1
内部的形状选择 `PagedAttention::eval_gpu`。任务4没有添加新内容
扩展函数。

Week 3 模式应该自动使用被支持的平板页路径
长预填. 简短的查询继续通过矢量页解码时间表进行.
两条路径都无法聚集密集的 K/V 声波 。

在连续跟踪中测量已完成的操作员。 迅速报告
范围、页面大小、批量大小、硬件、 prefill 吞吐量 decode 吞吐量
请求吞吐量, 峰值 KV 存储, 和逻辑 KV 复制量 :

```bash
pdm run bench-serving-progression --offline --repeats 4 \
  --model qwen3-4b --num-seqs 16 --batch-size 4 \
  --min-input-len 128 --max-input-len 1024 \
  --min-output-len 32 --max-output-len 128 --prefill-step 128 \
  --warmup 1 --cooldown-seconds 1 \
  --json-output benchmark_results/m4-pro-qwen3-4b-week3-serving-mlx-0.32.0.json
```

FlashAttention 期望会更重要,因为 prefill 增长. 它不应该
替换第4天 decode 排程 : 单键查询没有重用查询牌。

在已检查的M4 Pro追踪中,整个直页路径达到679.56
prefill tok/s, 41.88 产出 tok/s, 82.11 decode tok/s,和0.558请求/s.
相对于同一跟踪的密集服务,输出和请求吞吐量是:
28.7%以上, decode 吞吐量增加62.8%,高峰KV存储量增加47.4%。
低一点 这些是第3周累积的路径结果;提供的跟踪不
隔开第5天 prefill 从呼号、 直接 decode,或排程。

使用单独的8K静态扫描作为内核诊断在服务跟踪后进行.
它显示查询跳动何时开始抵消页面表的间接费用,但没有
量度请求转换、页面再利用或容量。 那个
[业绩附录](./appendix-performance.md) 记录匹配的服务
和长文测量。 长文 decode 仍为第4天矢量
内核工作量;不贷记a prefill 时间表a decode 增减。

```bash
pdm run bench-course-progression --offline --suite course \
  --variant week2 --variant week3 --variant mlx --model qwen3-4b \
  --input-len 8192 --output-len 2 --prefill-logits last \
  --warmup 1 --repeats 4 --cooldown-seconds 1 \
  --json-output benchmark_results/m4-pro-qwen3-4b-week3-8k-mlx-0.32.0.json
```

| 8K 静态检查站 | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| 第2个星期 | 323.26 | 17.62 |
| 第3周结束 | 424.14 | 24.96 |
| MLX | 594.21 | 25.24 |

整个第3周 prefill 路径比第2周快31.2%,达到71.4%。
页:1 MLX 在这个形状。 内容 decode 行是第4天矢量表,不是证据
用于平面 prefill 内核.

{{#include copyright.md}}
