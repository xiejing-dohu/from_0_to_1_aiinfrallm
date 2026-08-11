# 第 1 周第 4 天：RMSNorm 与多层感知机

我们将在第4天执行《公约》的两个重要组成部分。 Qwen3 变形器架构 : RMSNorm 和多层
perceptoron (MLP),又称feed-forward网络. RMSNorm 是一种常态技术,计算较少
比传统的层层正常化还要高 MLP在关注区块后应用非线性转换.

## 任务1:执行 `RMSNorm`

在这项任务中,我们将执行 `RMSNorm` 层。

```
src/tiny_llm/layer_norm.py
```

使用的第3天 `mx.fast.rms_norm` 直接这样,GQA的章节就可以继续专注于关注. 此任务执行
与可重复使用的图层相同的正常化规则。 从这里开始,变形器块, 最终模式正常化,和
Q/K 正常化路径可以使用 `RMSNorm` 执行。

**读数**

* [旋转平方层正常化](https://arxiv.org/abs/1910.07467)
* [Qwen3 mlx-lm(包括)层执行 RMSNorm)](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py) - 看到没? `RMSNorm`.


RMSNorm 定义为:

$$
y = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \text{weight}
$$

其中:

- `x` 是输入的亮度。
- `weight` 是学习到的缩放参数。
- `epsilon` (`eps`)是一个小常数,例如 `1e-5` 或 `1e-6`,用于数字稳定性。
- `mean(x^2)` 是方块元素沿着最后维度的平均值。

沿着输入的最后一个维度,独立对每个特性矢量应用正态. 将输入复制到 `float32`
用于正常计算,包括平均值,在原始值使用时保持精确 `float16` 或
`bfloat16`中输入 dtype 申请前 `weight`。这符合低精度
路径 MLX动作快 RMSNorm 内核:常态统计在 `float32`,而最后缩放
发生在模型中 dtype.

```
D is the embedding dimension.

x: N.. x D
weight: D
output: N.. x D
```

您可以通过运行测试您的执行 :

```bash
pdm run test --week 1 --day 4 -- -k task_1
```

## 任务2:执行 MLP 块

在这一任务中,我们将实施名为MLP的块 `Qwen3MLP`.

```
src/tiny_llm/qwen3_week1.py
```

原变形器在每个区块中都使用一个简单的位置明智的向导网络(FFN). 它由两个线性组成
在它们之间进行再LU激活的转换。

现代变形器建筑,包括 Qwen3,经常使用更先进的FFN变体. Qwen3 使用 SwiGLU,一个有门的线性
单位(GLU)变体。

普通FFN可以抽象为:

```plain
h = activation(W_up(x))
out = W_down(h)
```

GLU 保持相同的扩展- 当时的项目背面形状, 但添加了另一个投影, 将中间特性打开 。
`W_down`。这使 MLP 具有一种学习型的、依赖输入的方法来控制哪些中间通道很重要,而不是
仅将激活应用到 `W_up`.

SwiGLU是GLU的变体,用于: Qwen3:

```plain
u = W_up(x)
g = SiLU(W_gate(x))
out = W_down(g * u)
```

**读数**

- [注意是所有你需要的(译文,第3.3节,“Position-wise-Feed-Forward Networks”)](https://arxiv.org/abs/1706.03762)
- [GLU论文:用Gated革命网络进行语言建模](https://arxiv.org/pdf/1612.08083)
- [SiLU(Swish)激活功能](https://arxiv.org/pdf/1710.05941)
- [SwiGLU文件:GLU变体改进变形器](https://arxiv.org/abs/2002.05202v1)
- [PyTorch SiLU 文档](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html)
- [Qwen3 mlx-lm中的层执行(包括 MLP)](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py)

SwiGLU将一个GLU与SiLU(sigmoid线性单位)激活函数结合:

- GLU将输入的线性投影与输入的线性投影相连接,使用元素相乘来控制哪些特性
  通过。
- SiLU是一种平滑的,非摩诺活化功能. 与RELU不同,它没有零梯度区域横跨所有负数
  输入并可以产生负值的非零产出。

第一,执行 `silu` 输入 `basics.py`。它需要变形 `N.. x I` 并返回形状相同的变压器 :

$$
\text{SiLU}(x) = x * \text{sigmoid}(x) = \frac{x}{1 + e^{-x}}
$$
以数字稳定的方式计算 sigmoid 部分 :

```text
if x >= 0:
    sigmoid(x) = 1 / (1 + exp(-x))
else:
    sigmoid(x) = exp(x) / (1 + exp(x))
```

负分支在代数上相当于直接的sigmoid公式,但它可以防止 `exp(-x)` 从成为
`exp(large positive)` 何时 `x` 是一个巨大的负值。 在矢量代码中,首先计算 `z = exp(-abs(x))`启用
`z / (1 + z)` 负数投入和 `1 / (1 + z)` 否则。 不将负分支重写为
`1 - 1 / (1 + z)`:在低精度下,分数可以绕到 `1`,然后不正确生成零。

然后执行 `Qwen3MLP`. Qwen3MLP包含:

- 大门投影$W_{gate}$)
- 上升预测($W_{up}$)
- SiLU适用于门投影输出
- 激活门输出和向上投影输出的元素化产品
- 最后下降预测($W_{down}$)

这可以表现为:

$$
\text{MLP}(x) = W_{down}(\text{SiLU}(W_{gate}(x)) \odot W_{up}(x))
$$
地点 $\odot$ 表示元素的乘法。 Qwen3MLP的预测没有使用偏差.

```
N.. is zero or more dimensions for batches
E is hidden_size (embedding dimension of the model)
I is intermediate_size (dimension of the hidden layer in MLP)
L is the sequence length

input: N.. x L x E
w_gate: I x E
w_up: I x E
w_down: E x I
output: N.. x L x E
```

您可以通过运行测试您的执行 :

```bash
pdm run test --week 1 --day 4 -- -k task_2
```

最终,你应该能够通过今天的所有测试:

```bash
pdm run test --week 1 --day 4
```

{{#include copyright.md}}
