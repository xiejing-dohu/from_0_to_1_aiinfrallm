# 第 1 周第 3 天：分组查询注意力（GQA）

第3天,我们将执行 grouped-query attention (GQA) Qwen3 使用 GQA 来降低计算和内存成本
关键(K)和价值(V)预测。 内 multi-head attention (MHA),每个查询(Q)头都有对应的K和
V头. 随着GQA,Q头组共享K和V头. Multi-query attention (MQA)是每个儿童在下列情况下的特殊情况:
Q头共用单K/V头对.


**阅读**

- [GQA纸](https://arxiv.org/abs/2305.13245)
- [Qwen3 层,以mlx-lm计](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py)
- [PyTorch放大点产品注意度与 `enable_gqa=True`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [`torchtune.modules.MultiHeadAttention`](https://pytorch.org/torchtune/0.3/generated/torchtune.modules.MultiHeadAttention.html)

## 任务1:执行 `scaled_dot_product_attention_grouped`

您需要修改以下文件 :

```
src/tiny_llm/attention.py
```

在这项任务中,我们将实施分组式的点产品关注,这是全球质量评估的核心。

执行 `scaled_dot_product_attention_grouped` 输入 `src/tiny_llm/attention.py`。它类似于标准缩放点产品
注意,但它支持多个查询头,这是密钥/值头数的倍数.

主要过程与标准尺度的点产品关注相同. 不同的是K和V头是共享的
跨越多个Q头. 改为 `H_q` 分开的K和V头,有 `H` K和V头,每头由
`n_repeats = H_q // H` 查询头。

调整形状 `query`, `key`,以及 `value` 从而在 K 和 V 期间向各自组的查询头播放
矩阵乘法。

- 分开 `H` 和 `n_repeats` 维度 `query`.
- 增加尺寸1的尺寸 `n_repeats` 输入 `key` 和 `value` 他们每组人中都有播音员。

然后进行规模化的点产品注意:矩阵乘法,缩放,可选的遮罩, softmax和最后矩阵
乘论. 广播处理头部共享,不实现重复的K和V对称.

使用广播而不是重复K和V效率更高,因为它避免了生成相同数据的拷贝.

最后,将结果重塑为预期输出形状.

```
N.. is zero or more dimensions for batches
H_q is the number of query heads
H is the number of key/value heads (H_q must be divisible by H)
L is the query sequence length
S is the key/value sequence length
D is the head dimension

query: N.. x H_q x L x D
key: N.. x H x S x D
value: N.. x H x S x D
mask: N.. x H_q x L x S
output: N.. x H_q x L x D
```

除了分组标题外,此函数还支持不同的查询和键/值序列长度: Q使用长度 `L`,
K 和 V 使用长度 `S`.

您可以通过运行以下命令来测试您的执行 :

```bash
pdm run test --week 1 --day 3 -- -k task_1
```

## 任务2:因果面具

**阅读**

- [写 LLM 第九编 -- -- 因果关系](https://www.gilesthomas.com/2025/03/llm-from-scratch-9-causal-attention)

在这项任务中,我们将在分组关注的基础上增加因果掩盖。

致病面具使注意力无法阅读未来的符号. 何时 `mask` 设置为字符串 `"causal"`,应用因果关系
戴着面具。

添加剂因果面具有形状 `(L, S)`时, `L` 是查询序列长度,并且 `S` 是键/值序列长度。
允许的职位包括0个,蒙面的职位包括 `-inf`时 `S` 大于 `L`转换对角
`S - L` 以便查询与最后 `L` 键/值序列中的位置。 例如,如果 `L = 3`
和 `S = 5`这个面具是:

```
0   0   0   -inf -inf
0   0   0   0    -inf
0   0   0   0    0
```

执行 `causal_mask` 输入 `src/tiny_llm/attention.py`,然后在 `scaled_dot_product_attention_grouped`。注意:
我们转动对角 `L != S` 不同于一些关注API的默认行为.

您可以通过运行以下命令来测试您的执行 :

```bash
pdm run test --week 1 --day 3 -- -k task_2
```

## 任务3: Qwen3 分组查询注意

在这项任务中,我们将执行 Qwen3's grouped-query attention。修改以下文件:

```
src/tiny_llm/qwen3_week1.py
```

`Qwen3MultiHeadAttention` 执行注意 Qwen3。遵循此伪代码:

```
x: B, L, E
q = linear(x, wq) -> B, L, H_q, D
k = linear(x, wk) -> B, L, H, D
v = linear(x, wv) -> B, L, H, D
q = rms_norm(q, q_norm)
k = rms_norm(k, k_norm)
q = rope(q, offset=slice(0, L))
k = rope(k, offset=slice(0, L))
(transpose as needed)
x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, H_q, L, D  # use float32
(transpose as needed)
x = linear(x, wo) -> B, L, E
```

Qwen3 注意没有Q/K/V投影偏差,适用 RMSNorm 在每个 Q 和 K 头之前 RoPE我们将执行
重用时 `RMSNorm` 第4天的一层,你叫 `mx.fast.rms_norm` 直接用于 `q_norm` 和 `k_norm` 今时. 使用
非传统 RoPE.

您可以通过运行以下命令来测试您的执行 :

```bash
pdm run test --week 1 --day 3 -- -k task_3
```

最终,你应该能够通过今天的所有测试:

```bash
pdm run test --week 1 --day 3
```

{{#include copyright.md}}
