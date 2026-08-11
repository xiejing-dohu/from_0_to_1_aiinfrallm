# 🚧 第 2 周第 5 天：融合 Decode Attention

> **现状:实验。** 见
> [第2周核查矩阵](./week2-overview.md#verification-status) (单位:千美元)
> 不断测试、在当地测量和仍在审查中的内容。

第4天证据证实 保险丝已经解除
模型内核减小了重复的指针集群。 线性预测仍然存在
很重要,但是他们的操作员的耐久性 已经接近外部
分母,而注意则是通过缓存测量的下一个可移动缺口
上下文 `S <= 256`. 使用第1周进行较长文字测量 Python 倒计时
超过 256 的缓存;通过 256 的每个检查上下文都使用优化路径. 期间
单项请求 decode,查询长度通常是一个,而缓存的密钥/值
序列由一个增长 token 一次来 第1周作为矩阵表示注意
乘法 蒙面术 softmax,和另一个矩阵乘法。 这是
表示与 `mlx.core`,但它实现了 完整的分数和概率行。

先写一个 Python `mlx.core` 组合以保留等式,然后替换其
mat和 mat softmax 带有在线软件 Metal 内核在你的解决办法。
在决定是否保留发送之前,先测量完整的模式。 那个
内核不调用 `mx.matmul` 或 MLX 提供的
缩放产品注意
执行; MLX 仍然提供数组、流、缓冲和扩展
发送。

## 任务1:保护接口

修改 `scaled_dot_product_attention` 输入
`src/tiny_llm/week2_kernels.py`。保持此可读功能作为预告和
回落; 任务 2 修改单独的 `decode_attention_custom` 进入点。

执行 `scaled_dot_product_attention` 输入 `week2_kernels.py` 与这些
模型形状 :

```plain
query: B, H_q,  L, D
key:   B, H_kv, S, D
value: B, H_kv, S, D
out:   B, H_q,  L, D
```

验证 `H_q` 由 `H_kv`平整批量和头尺寸
以 :

```plain
kv_head = query_head / (H_q / H_kv)
```

将清晰的口罩规范化 `B * H_q, L, S`。通过一个因果标志
内核可以跳过未来位置而无需构建一个因果-mask测距器.

作为一名 Python 中间步骤,将查询头重塑为 `H_kv` 团体和a
重复维度。 然后广播将几个查询头和一个KV头对齐
而不在物理上重复键和值值。 速成分数
softmax,并明确列出加权值产品。 使用此表单为
正确性甲骨文和衰竭,而不是作为完成的优化路径:
matmuls是MLX提供的操作器执行.

## 任务2:执行 Online Softmax 输入 Metal

修改 `tiny_llm_ext::decode_attention`,
`Week2DecodeAttention::eval_cpu`,以及 `Week2DecodeAttention::eval_gpu` 输入
`src/extensions/src/week2_kernels.cpp`,则 `week2_decode_attention` 函数
输入 `src/extensions/src/week2_kernels.metal`,以及 `decode_attention_custom` 输入
`src/tiny_llm/week2_kernels.py`。启动声明、约束、来源
树枝, Metal 文件,且 CMake 注册已经存在并标签为周
2 第5天;替换这些已关闭的机构,而不是增加新的名称。

展览 `decode_attention_custom` 联 合 国 Metal 执行。 缓存
在缓存行走之前缩放登记簿中的查询片段; 再次加载
每个关键位置都可以避免。 指派32个32行 SIMD 各组
128-192 上的查询行 token 基准。 每个小组每32次访问缓存
位置; 在一组内:

1. 每一车道乘以一个固定空格的查询和密钥值子集。
2. `simd_sum` 把这些部分点产品合并成一个分数.
3. 采用比例表、可选面具和因果核对。
4. 更新运行上限, softmax 分母和加权值
   积者.

在线更新是:

```plain
new_max = max(running_max, score)
old_factor = exp(running_max - new_max)
score_factor = exp(score - new_max)
denominator = denominator * old_factor + score_factor
accumulator = accumulator * old_factor + score_factor * value
```

在它最后的缓存位置之后,每个组写出其部分最大,
分母,并值累积器到线组内存。 第一个 SIMD 组
计算通用的最大值和重度系数。 一个线程计算最后
分母,然后是第一个 `D` 每个线程都结合了一个输出维度。 这个
将最终的减值与头部维度平行。
切换最大值给稳定 softmax 不存储全部 `S` 分数或
概率

这可以消除两个大中间体和几个发送边界
第1周图. 随着环境的增长,它特别相关:避免得分和
概率阈值与 `L * S`时 decode 仅需要
最终 `D`-每个查询头都有内容结果。

装入和存储 BF16 直接,但积聚点产品,
softmax 状态,和浮点32中的加权值。 铸造整张Q,K,和V连载器
内核外创建额外的调度和内存流量; 执行
登记册中的转换可以避免这一费用。

使用 `fast::exp` 缩放因子并计算
在应用到分母和所有价值维度之前,一个系数。
这些想法也出现在生产矢量-注意内核中,包括: MLX's
SDPA消息来源. 您的内核重新执行算法和排程
其本身 Metal 代码; 不包括或即时 MLX 内核.

### 排程实验

比较八十六和三十二 SIMD 持有时带有 Qwen3-4B 的组合
上下文已经固定。 组数是一个工作量参数,而不是
通用常数:更多群体揭露平行得分工作,但消耗更多
线程和线程组内存。 记录同步运算符和
每个调度的完整模型结果,然后在
上下文长度变化。

## 任务3:整合和措施

修改 `Qwen3MultiHeadAttention.__call__` 输入
`src/tiny_llm/qwen3_week2.py` 以应用测量的调度器。 保留
`scaled_dot_product_attention` 作为明显的倒置和呼叫
`decode_attention_custom` 仅在受援区域内。

路线短小的,短短的文字 第2周通过 Metal
执行。 调回总部 Python `mlx.core` 缓存时的构成
上下文超过测量到的交叉; 以 128 个符号胜出的时间表
不应强迫2 048个符号。 保留 Python 人员构成
测试和烧伤。 第3周之后,将这种重现与页K/V和
用于 SIMD 矩阵牌 FlashAttention; prefill 属于不同的工作量,其中:
查询和上下文长度都很大。

设置混凝土调度守卫: 使用您的 Metal 仅当查询长度为
最多2个,缓存的上下文长度最多256个。 否则使用
Python 分组注意路径 。 保持这个条件 在模型呼叫站点,所以
基准操作范围仍可审查,而不是成为
隐藏在 Metal 内核.

保持任意密集,每要求口罩 Python 模式路径。 那个
原始人仍然接受清晰的面具 所以它的算术合同可以
测试过,但第2周调度警卫只选择自定义内核
`None` 或 `"causal"`. 第一次连续打击中出现明确的口罩
练习,而普通的单一请求 decode 没有面具。 第3周替换
密密的批量口罩,上面刻有调用元数据,而不是隐藏
在这个重点突出的模型路径中的绩效政策.

```bash
pdm run build-ext
pdm run test --week 2 --day 5
```

测试分组- query 头映射、 输出形状、 因果行为和清晰
口罩对面 Python 第1周执行。 参考套房使用
Qwen's `D = 128`,查询长度1和8,GQA比率1和4,以及缓存上下文
`1, 31, 32, 127, 128, 129, 255, 256`也检查模型的两侧
`L <= 2` 和 `S <= 256` 派出警卫 使用宽容,因为 online softmax
更改浮点还原顺序。

网格的正确性不能证明一个32SIMD固定的组表
高效。 在背景1,8和31中,它的许多1,024线程没有得分
处理位置。 运行相同的真实形状操作员扫描每个目标
保留时间表前的机器 :

```bash
for context in 1 31 32 127 128 129 255 256; do
  pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
    --section attention --context "${context}" \
    --query-length 1 --gqa-ratio 4 --attention-mask none
done

for context in 8 31 32 127 128 129 255 256; do
  pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
    --section attention --context "${context}" \
    --query-length 8 --gqa-ratio 4 --attention-mask causal
done
```

重复代表点与 `--gqa-ratio 1` 和
`--attention-mask explicit`将 M1 和 M4 的结果作为单独记录;a
M1 CI跑车的正确性不是M4交叉的证据
适用。

运行前一个检查点, 并使用您的新调度程序
原本相同的设置 :

```bash
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint swiglu --model qwen3-4b \
  --num-seqs 1 --min-input-len 32 --max-input-len 32 \
  --min-output-len 97 --max-output-len 97 --warmup 2 \
  --prefill-logits last

pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint decode-attention --model qwen3-4b \
  --num-seqs 1 --min-input-len 32 --max-input-len 32 \
  --min-output-len 97 --max-output-len 97 --warmup 2 \
  --prefill-logits last
```

Prefill 生成第一个 token,所以96时间 decode 呼叫增长缓存
从 `S=33` 结束 `S=128`所有人都在定制的调度哨里
你的解决方案回到了准确的 Python 第1周外的组成
验证范围。

## 基准分析:核查 Prefill 预测 下一波特伦克

分别测量注意操作员和累计检查点。 那个
第一个进展是与此匹配的短文本接受测试
界内核. 第二个保留了固定的周2分母:它的128个token
prefill 保留在查询长度保护之外,而计时为单键 decode
步骤 `S=129` 结束 `S=256` 输入当前上下文保护 :

```bash
pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
  --section attention --context 32 --context 128 --context 160 \
  --context 192 --context 256 --context-repeats 6 \
  --json-output benchmark_results/week2-attention-context-sweep.json

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-swiglu --variant week2-decode-attention --variant mlx \
  --model qwen3-4b --input-len 32 --output-len 97 --warmup 2 \
  --prefill-logits last

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-swiglu --variant week2-decode-attention --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last

```

在背景32、128、160、192和256中重复注意微观基准,以及
将上下文扫描到短文旁边
`swiglu`/`decode-attention` 模组列。
中间点显示自定义内核是否有有用的测量
交叉而不是假设一个终点适用于每个上下文。
如果重复的新鲜进程短文本运行不拒绝自定义发送
改进,即使孤立的内核看起来更快。 如果操作员只赢了
在有限的上下文范围内, 编码在发送中测量交叉
警卫

> **选择性貌相证据.** Decode 和 prefill 内核组结果可以
> 解释工作量如何分配时间,但它们是参考证据,
> 该检查站不需要产出。

M4 Pro上检查的Quen3-4B扫描使用了6个前向/反向上下文通过,
旋转每个执行命令,记录所有60个样本
执行和通过:

| 背景情况 | Python 参考文献 | Metal | MLX | Metal 加速 |
|---:|---:|---:|---:|---:|
| 32 | 我们143.0 | 125.7 我们 | 116.3我们 | 1.138x |
| 128 | 我们149.3 | 136.3我们 | 我们 | 1.095x |
| 160 | 我们151.2 | 140.1 我们 | 我们120.9 | 1.079x |
| 192 | 我们154.0 | 我们143.9 | 121.9 我们 | 1.071x |
| 256 | 158.0我们 | 我们150.7 | 122.8我们 | 1.048x |

那个 Metal 路径通过256的每个测量点获胜,所以256是最大的
有证据的上下文 那个 Python `mlx.core` 路径仍然是政策之外
范围;不要将最终4.8%的运算符胜出到更长的缓存。 生下来的
记录,包括准确的源 SHA,模型配置, MLX 和 mlx-lm 键
各个版本, Metal 编译器版本,设备信息,执行命令,样本,
中间数,为
`benchmark_results/m4-pro-qwen3-4b-week2-attention-context-sweep-mlx-0.32.0.json`.

生产边界扫荡的上下文为128,选用Qwen3-4B的4:1 GQA
比例,以及平衡的L1/L2/L4/L8 6次通过。 它使用因果关系
每个查询长度: `L=1` 该掩码允许整个现有缓存,
相当于未装模作样的单键 decode时 `L>1` 措施因果
多键块 。 每张通行证还轮流执行三项执行命令,
保留每个样本 :

| 查询长度 | Python 参考文献 | Metal | MLX | Metal 加速 | 传球获胜 |
|---:|---:|---:|---:|---:|---:|
| 1 | 244.4 我们 | 我们213.1 | 我们155.9 | 1.147x | 6/6 |
| 2 | 341.4我们 | 258.8我们 | 我们185.3 | 1.319x | 6/6 |
| 4 | 322.7 我们 | 297.3 我们 | 197.4我们 | 1.085x | 4/6 |
| 8 | 377.7 我们 | 我们491.5 | 我们290.6 | 0.768x | 0/6 |

L4的总和中位数改善,但损失了6个平衡的通行证中的2个. L2为
最大的重复胜利, 所以调度守卫保持保守
时间 `L <= 2`; L4 和 L8 使用 Python 路径。 重现记录的扫荡方式:

```bash
pdm run bench-week2-operators --solution tiny_llm_ref --model qwen3-4b \
  --section attention --context 128 \
  --query-length 1 --query-length 2 --query-length 4 --query-length 8 \
  --gqa-ratio 4 --attention-mask causal --context-repeats 6 \
  --warmup 12 --iterations 60 \
  --json-output benchmark_results/week2-attention-query-sweep.json
```

检查的原始记录是
`benchmark_results/m4-pro-qwen3-4b-week2-attention-query-sweep-mlx-0.32.0.json`.

在定时 `128/129` 工作量, prefill 已经 `L=128` 并使用 Python 路径。
这是第一次 decode 调用新附件 token 在警卫看到之前 `S=129`;
单键 decode 通话 `S=256` 因此使用自定义路径。 保留
与短文接受程序分开的固定工作量。 继续
正确度测试合格后第6天 直接来源的痕迹证明了
限制注意力发送及其倒置, 重复的短文运行
和固定的, `128/129` 控制证实 prefill 不变,且
仍然通过Day 3的矩阵形状的投影路径.

> **选择性貌相证据.** 那个
> [参考检查站](./appendix-performance.md#day-5-fused-decode-attention)
> 对上下文扫描、短文本模型三角洲和固定工作负荷
> 带有单独的控制 prefill 归属。 这种归属解释了为什么
> 课程目标矩阵形状的下一个预测;这不是实现
> 第6天。

{{#include copyright.md}}
