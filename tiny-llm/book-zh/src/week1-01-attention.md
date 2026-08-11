# 第 1 周第 1 天：注意力与多头注意力

在第一天,我们将执行基本关注和 multi-head attention。注意层处理输入序列和
在生产每项产出时,权衡其不同位置的相关性。 注意是变形器的关键构件
模特儿们

[📚 阅读：Transformer 架构](https://huggingface.co/learn/llm-course/chapter1/6)

我们用 Qwen3,用于文本生成。 模型需要一个序列 token 身份证,地图 嵌入,
并生成下一个对数 token 在每一个序列位置。 生成循环将使用最终位置
登录来选择下一个 token 身份证

[📚读: LLM 推理 Decode 阶段](https://huggingface.co/learn/llm-course/chapter1/8)

注意层需要查询、键和值。 在基本实施中,这三项措施的形状相同:
`N.. x L x D`.

`N..` 表示零或更多批量尺寸。 每批货中 `L` 是序列长度和 `D` 嵌入为
一个注意头的维度。

例如,头尺寸为512的1 024个符号的序列由形状的拉伸表示
`N.. x 1024 x 512`.

## 任务1:执行 `scaled_dot_product_attention_simple`

在这项任务中,我们将扩大对点产品的关注。 我们假设输入的分数Q,K,和V有相同的
形状。 之后的章节将引入输入形状不同的关注变体.

```
src/tiny_llm/attention.py
```

**读数**

* [附加说明的变形器](https://nlp.seas.harvard.edu/annotated-transformer/)
* [PyTorch 缩放点产品注意 API](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (掌声) `enable_gqa=False`,假设 dim k=dim v=dim q 和 H k=H v=H q)
* [MLX 缩放点产品注意 API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html) (ASume dim k=dim v=dim q和H k=H v=H q) ).
* [注意只是你需要的](https://arxiv.org/abs/1706.03762)

执行 `scaled_dot_product_attention_simple` 使用以下公式。 该函数接收查询、 密钥和值等分数
外形相同,外加可选添加剂面具 `M`.

$$
  \text{Attention} = \text{softmax}(\frac{QK^T}{\sqrt{d_k}} + M)V
$$

在这里, $\frac{1}{\sqrt{d_k}}$ 是默认比例系数。 呼叫者可能提供不同的比例系数。

```
L is seq_len, in PyTorch API it's S (source len)
D is head_dim

key: N.. x L x D
value: N.. x L x D
query: N.. x L x D
output: N.. x L x D
scale = 1/sqrt(D) if not specified
```

你可以用这个 MLX's `softmax`我们将在第2周重新讨论低级行动。

当调用此函数时 multi-head attention,声波通常会有这些形状:

```
key: 1 x H x L x D
value: 1 x H x L x D
query: 1 x H x L x D
output: 1 x H x L x D
mask: 1 x H x L x L
```

该功能本身在最后两个层面运作,必须支持任何数量的主要批量层面。 面罩
只需要一个能向注意分数形状广播的形状。

完成此项任务后,应当能够通过下列测试: 1.

```
pdm run test --week 1 --day 1 -- -k task_1
```

## 任务2:执行 `SimpleMultiHeadAttention`

在这项任务中,我们将执行 multi-head attention 层。

```
src/tiny_llm/attention.py
```

**读数**

* [附加说明的变形器](https://nlp.seas.harvard.edu/annotated-transformer/)
* [PyTorch 多头 API](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MultiheadAttention.html) (ASume dim k=dim v=dim q和H k=H v=H q) ).
* [MLX 多头保留 API](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MultiHeadAttention.html) (ASume dim k=dim v=dim q和H k=H v=H q) ).
* [Illustrated GPT-2(视觉变形语言模型)](https://jalammar.github.io/illustrated-gpt2) 帮助您更好地了解什么是关键、价值和查询。

执行 `SimpleMultiHeadAttention`。该层项目以 Q、K 和 V 批次查询、键和值向量
重量矩阵,然后将预测值从任务1转到注意函数。 最后,它应用了输出预测
O.

第一,执行 `linear` 函数在 `basics.py`。它需要变形 `N.. x I`,形状的加权矩阵
`O x I`,和形状的可选偏差矢量 `O`。其输出有形状 `N.. x O`时, `I` 是输入维度,并且
`O` 是输出维度。

为: `SimpleMultiHeadAttention`,输入分数 `query`, `key`,以及 `value` 有形状 `N x L x E`时, `E` 这是
嵌入一个维度 token. Q、K和V 预测每个地图 `E` 改为 `H x D`: `H` 头,每个都有尺寸
`D`将最后投影维度重新塑造成单独的 `H` 和 `D` 维度。

你现在的形状变形了 `N x L x H x D` 每个投影。 注意前,将每个移到
`N x H x L x D`.

- 将每个注意头作为独立的批次处理,以便分别计算每个注意头的注意
  跨序列维度 `L`.
- 离开 `H` 之后 `L` 会导致矩阵乘法将头和顺序维相混合。 每个人必须出席
  仅改为 token 其子空间内的关系。

注意功能每个头产生一个输出. 将结果转换回 `N x L x H x D`,重塑为
`N x L x (H x D)`,并应用输出投影。

```
E is hidden_size or embed_dim or dims or model_dim
H is num_heads
D is head_dim
L is seq_len, in PyTorch API it's S (source len)

w_q/w_k/w_v: (H x D) x E
output/input: N x L x E
w_o: E x (H x D)
```

任务结束时,应当能够通过下列测试: 1.

```
pdm run test --week 1 --day 1 -- -k task_2
```

您可以使用下列方式进行今天的所有测试:

```
pdm run test --week 1 --day 1
```

{{#include copyright.md}}
