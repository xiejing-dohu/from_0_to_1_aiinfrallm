# 第 1 周第 6 天：生成响应——Prefill 与 Decode

第6天,我们将实施反应生成 LLM 聊天员 执行时间很短,但效果很大
代码。 使用本章整合并调试完整的第1周模式.

## 任务1:执行 `simple_generate`

```
src/tiny_llm/generate.py
```

`simple_generate` 使用一个模型、 标注器、 快速和可选的采样器, 然后将生成的响应流到标准
输出。 世代分两个阶段: prefill 和 decode.

首先,执行巢 `_step` 函数。 它需要一维阵列 token 身份证 加上批量尺寸
然后把结果传给模特儿 模型返回每个序列位置的词汇表上的非正常日志 。

```
y: S (before adding a batch dimension)
model input: 1 x S
output_logits: 1 x S x vocab_size
```

你只需要最后一个 token下一个决定的日志 token。因此,您需要选择最后一个 token'对数'
从输出日志。

```
logits = output_logits[:, -1, :]
```

你可以把这些日志 变成日志概率 和日志和输出的诡计 这种正常化没有改变
成果 `argmax`,但第7天引入的采样器预计日志概率. 若为 `sampler` 这是 `None`编辑
`mx.argmax` 沿着最后的词汇维度。 否则,将日志概率传递给 `sampler`选择
最高评分 token 每一步都称为贪婪的解码。

- 📚 [Log-Sum-Exp 技巧](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
- [大语言模型中的解析策略](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html)
- [开关定义](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer)

与 `_step` 完成,执行其余部分 `simple_generate`。开始将提示编码为一维 token
阵列为 `tokenizer.encode`.

在循环中生成令牌, 直到模型释放 `tokenizer.eos_token_id`添加每个新的 token 页:1 token 数组这样
下一个型号呼叫收到完整的序列。 将非 EOS 输出符号输入到 `tokenizer.detokenizer`,并打印每个
新的文本段已可用。

向《公约》提供序列的例子 `_step` 函数如下:

```
tokenized_prompt: [1, 2, 3, 4, 5, 6]
prefill: _step(model, [1, 2, 3, 4, 5, 6]) # returns 7
decode: _step(model, [1, 2, 3, 4, 5, 6, 7]) # returns 8
decode: _step(model, [1, 2, 3, 4, 5, 6, 7, 8]) # returns 9
...
```

在第二周,我们将用密钥值缓存加速解码,这样模型就不会重算整个序列
在每一个步骤。

您可以通过运行以下命令来测试您的执行 :

```bash
# Start with the default 0.6B model.
hf download Qwen/Qwen3-0.6B-MLX-4bit
pdm run main --solution tiny_llm --loader week1 --model qwen3-0.6b \
  --prompt "Give me a short introduction to large language model"

# If downloaded, you can also try the larger models.
pdm run main --solution tiny_llm --loader week1 --model qwen3-1.7b \
  --prompt "Give me a short introduction to large language model"
pdm run main --solution tiny_llm --loader week1 --model qwen3-4b \
  --prompt "Give me a short introduction to large language model"
```

每个命令都应该对大语言模型作出合理的解释. 替换 `--solution tiny_llm` 与
`--solution ref` 以运行参考解决方案。

{{#include copyright.md}}
