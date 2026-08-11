# 第 1 周第 5 天：Qwen3 模型

第5天,我们将把前几章的内容合并为完整的 Qwen3 型号。

模型级测试需要相应的模型文件. 以默认的 0. 6B 模式开始; 下载更大的模式
除非你也想测试一下

```bash
hf download Qwen/Qwen3-0.6B-MLX-4bit
# Optional larger models:
hf download Qwen/Qwen3-1.7B-MLX-4bit
hf download Qwen/Qwen3-4B-MLX-4bit
```

需要不可用模型的测试将被跳过 。

## 任务1:执行 `Qwen3TransformerBlock`

```
src/tiny_llm/qwen3_week1.py
```

**读数**

- [变形板简化解释](https://medium.com/@akhileshkapse/a-simplified-explanation-of-the-transformer-block-must-read-blog-for-nlp-enthusiasts-12ef240a62ac)
- [注意是所有你需要的](https://arxiv.org/pdf/1706.03762)

Qwen3 使用以下变形器块结构:

```
  input
/ |
| input_layernorm (RMSNorm)
| |
| Qwen3MultiHeadAttention
\ |
  Add (residual)
/ |
| post_attention_layernorm (RMSNorm)
| |
| MLP
\ |
  Add (residual)
  |
output
```

以 :

```bash
pdm run test --week 1 --day 5 -- -k task_1
```

## 任务2:执行 `Embedding`

```
src/tiny_llm/embedding.py
```

**读数**

- [LLM 嵌入式解释:视觉和直观指南](https://huggingface.co/spaces/hesamation/primer-llm-embedding)

嵌入层图 token 长度向量的ID( 插入符) `embedding_dim`。在此任务中,您将执行
那个搜寻行动

```
Embedding::__call__
weight: vocab_size x embedding_dim
Input: N.. (tokens)
Output: N.. x embedding_dim (vectors)
```

这可以通过数组索引执行.

当输入和输出嵌入被绑定时, Qwen3 也使用嵌入量作为隐藏向量的线性投影
返回词汇日志。

```
Embedding::as_linear
weight: vocab_size x embedding_dim
Input: N.. x embedding_dim
Output: N.. x vocab_size
```

以 :

```bash
# This task's tests use the 0.6B model and tokenizer.
hf download Qwen/Qwen3-0.6B-MLX-4bit
pdm run test --week 1 --day 5 -- -k task_2
```

## 任务3:执行 `Qwen3ModelWeek1`

现在我们已经建立了所有 Qwen3 组成部分,我们可以执行 `Qwen3ModelWeek1`.

```
src/tiny_llm/qwen3_week1.py
```

您不会执行从 lomor 文件读取模型参数的过程 。 相反,装入模型 `mlx_lm`,
然后把它的参数转移到我们的执行中。 那个 `Qwen3ModelWeek1` 因此,建筑师接受 MLX 型号。

那个 Qwen3 模型有以下几层:

```
input
| (tokens: N..)
Embedding
| (N.. x hidden_size); note that hidden_size == embedding_dim
Qwen3TransformerBlock
| (N.. x hidden_size)
Qwen3TransformerBlock
| (N.. x hidden_size)
...
|
RMSNorm 
| (N.. x hidden_size)
Embedding.as_linear OR linear (lm_head)
| (N.. x vocab_size)
output
```

读取层数、 隐藏大小、 头尺寸和其他配置值 `mlx_model.args`,其类型
由[`ModelArgs`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py)。负载重量
可通过 `mlx_model.model`; 使用 Qwen3 执行和元数据模型,以确定相应的
层名。

到此为止,你已经执行了 `RMSNorm`替换临时第3天的呼叫 `mx.fast.rms_norm` 与
`RMSNorm(head_dim, q_norm, eps=...)` 和 `RMSNorm(head_dim, k_norm, eps=...)`。它们执行相同的公式;内置公式
发出呼吁只是为了使全球质量评估的一章侧重于关注。

不同 Qwen3 模型变体以不同的方式将隐藏矢量映射回词汇日志. 有的将输入和
输出嵌入和使用 `Embedding.as_linear`; 其他人有一个单独的 `lm_head` 线性层。 选择策略
`mlx_model.args.tie_word_embeddings`: 如果是的话 `True`编辑 `Embedding.as_linear`; 否则, 装入和使用 `lm_head`.

模型需要一个序列 token 每个序列位置的ID和返回非正常的日志 。 在第六天,我们将
使用最后位置的日志选择下一个 token 并生成一个响应。

那个 MLX 本课程中使用的模型有定量权重. 逐个定号
线性或嵌入层后再装入 tiny-llm 通过使用所提供的
`quantize.dequantize_linear` 函数,然后将可读的周 1 重存储为
BF16模式激活和层层输出应保持不变 BF16可读
注意或正常化表达式可能计算为: FP32 为了稳定,但它
必须将其模型化的结果重新投向 BF16.

过 `mask="causal"` 到每个变形板块。 对于单键序列,面具没有作用;对于更长的序列,
它使每个职位都无法关注未来的迹象。

以 :

```bash
# Download each model you want to test. Missing models are skipped.
hf download Qwen/Qwen3-0.6B-MLX-4bit
hf download Qwen/Qwen3-1.7B-MLX-4bit
hf download Qwen/Qwen3-4B-MLX-4bit
pdm run test --week 1 --day 5 -- -k task_3
```

最终,你应该能够通过今天的所有测试:

```bash
pdm run test --week 1 --day 5
```

{{#include copyright.md}}
