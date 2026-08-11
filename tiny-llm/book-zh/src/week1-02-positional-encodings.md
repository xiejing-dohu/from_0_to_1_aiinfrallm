# 第 1 周第 2 天：位置编码与 RoPE

第2天,我们将执行 位置编码 Qwen3:旋转位置编码(RoPE) ). 变形器需要
一种代表每一个 token'位置在序列中. Qwen3 应用 RoPE 键向量
multi-head attention 层。

**读数**

- [你可以设计最先进的位置编码](https://huggingface.co/blog/designing-positional-encoding)
- [ Rofer: 带有旋转器位置编码的增强变形器](https://arxiv.org/pdf/2104.09864)

## 任务1:执行传统旋转器定位编码

您需要修改以下文件 :

```
src/tiny_llm/positional_encoding.py
```

传统 RoPE,如在读数中描述的那样,位置编码独立应用于每个查询头
和键向量。 您可以在初始化时预先计算频率 。 `RoPE` 课。

若为 `offset` 未提供,则应用职位0至 `L - 1` 输入序列。 否则,从
供应的切片。 例如,与 `offset=slice(5, 10)`,输入序列必须具有长度 5,并且其第一个 token
5号位置使用频率。

一周,你只需要支持 `offset=None` 单曲 `slice`我们将执行 `list[slice]` 连续
稍后进行批量处理。 现在,假设每批物品使用相同的冲抵。

```
x: (N, L, H, D)
cos/sin_freqs: (MAX_SEQ_LEN, D // 2)
```

传统 RoPE 沿着头维解释相邻值 `D` 作为复数对。 若为 `D = 8`,则 `x[0]`
和 `x[1]` 形成一对, `x[2]` 和 `x[3]` 组成另一个,等等。 成对的两个值使用相同的频率
`cos_freqs` 和 `sin_freqs`.

在实践中, `D` 可能是偶数或奇数。 如果是奇数,则最终值没有伴侣,通常保持不变。 为:
简便,执行需要 `D` 变得平和。

```
output[0] = x[0] * cos_freqs[0] + x[1] * -sin_freqs[0]
output[1] = x[0] * sin_freqs[0] + x[1] * cos_freqs[0]
output[2] = x[2] * cos_freqs[1] + x[3] * -sin_freqs[1]
output[3] = x[2] * sin_freqs[1] + x[3] * cos_freqs[1]
...and so on
```

您可以通过重塑执行此操作 `x` 改为 `(N, L, H, D // 2, 2)` ,并将公式应用到每对。

**读数**

- [PyTorch 扶轮转动装置 API](https://pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html)
- [MLX 实施 RoPE 在自定义之前 metal 内核执行](https://github.com/ml-explore/mlx/pull/676/files)

您可以通过运行以下命令来测试您的执行 :

```
pdm run test --week 1 --day 2 -- -k task_1
```

## 任务2:执行非传统 `RoPE`

Qwen3 采用非传统安排 RoPE 双人 将头尺寸分为两半,然后对齐
从两半的值。 让 `x1 = x[..., :HALF_DIM]` 和 `x2 = x[..., HALF_DIM:]`.

```
output[0] = x1[0] * cos_freqs[0] + x2[0] * -sin_freqs[0]
output[HALF_DIM] = x1[0] * sin_freqs[0] + x2[0] * cos_freqs[0]
output[1] = x1[1] * cos_freqs[1] + x2[1] * -sin_freqs[1]
output[HALF_DIM + 1] = x1[1] * sin_freqs[1] + x2[1] * cos_freqs[1]
...and so on
```

通过选择第一个和第二个半句法执行此表单 `x` 直接使用旋转和调制
结果。

**读数**

- [vLLM 执行情况 RoPE](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding)

您可以通过运行以下命令来测试您的执行 :

```
pdm run test --week 1 --day 2 -- -k task_2
```

最终,你应该能够通过今天的所有测试:

```
pdm run test --week 1 --day 2
```

{{#include copyright.md}}
