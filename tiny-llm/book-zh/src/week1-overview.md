# 第 1 周：从矩阵乘法到文本

本周,我们将从基本阵列和矩阵操作开始,并使用它们转动 Qwen3 模型参数输入模型
生成文本。 我们将实施神经网络层 Qwen3 与 MLX阵列 API 。

我们用 `Qwen/Qwen3-0.6B-MLX-4bit`。课程模式使用 BF16 重量和
默认激活, 因此在尝试更大的前先使用 0. 6B 模式 Qwen3
模特儿们 所需的模式路径运行于 GPU小型操作装置可能
使用不同的一个 dtype 作为可读的正确性参考;它们没有定义
模型存储 dtype.

微量敏感操作可促进计算 FP32 并铸造
结果返回到 BF16。第1周有利于可读数组表达式,即使这样
意味着实现 FP32 中间. 第2周取代满分
用内核将模型大小的存储保存在 BF16 并累积到
FP32 登记册。

## 我们的掩护

- 请注意, multi-head attention, grouped-query attention,以及 multi-query attention
- 定位编码和 RoPE
- 使用 `mx.fast.rms_norm` 完成 Qwen3 的逐头 Q/K 归一化，然后自行实现 RMSNorm
- 实施MLP,将注意力部分结合起来,建立完整的变形器模型
- 装弹 Qwen3 模型参数和生成文本

## 我们不会覆盖的

为了让旅程尽可能有趣,我们现在将略去一些事情:

- 量化与反量化的内部实现。这些内容将在第 2 周介绍；目前先使用提供的辅助函数
  将 Qwen3 权重,然后传给我们层层的执行。
- 低水平实施诸如以下行动: softmax,表示,和对数. 这些操作很简单
  足够使用 MLX 版本并不减损学习目标。
- token化。 我们用 `mlx_lm` 而不是从头开始执行
- 解码模型加权文件。 我们用 `mlx_lm` 以加载模型,然后将其重量转移到我们的层执行中。

## 基本矩阵 API

MLX's Python API 用于为NumPy用户熟悉。 如果您是新编程, 请从
[NumPy:初学者的绝对基础](https://numpy.org/doc/stable/user/absolute_beginners.html).

也可以提及[MLX 业务 API](https://ml-explore.github.io/mlx/build/html/python/ops.html#operations)
更多细节。

## Qwen3 模型

你们可以跑 Qwen3 与 MLX 或 vLLM。下面的阅读提供了我们将要建立的背景。 到周末的时候
您将可以使用 Qwen3 作为生成文本的因果语言模型。

参考执行 Qwen3 现存于Hugging Face Transformers, vLLM,还有mlx -lm. 用它们来探索
这个模型的内部信息 并且把它们和本周的实施相比较

**读数**

- [Qwen3:思考更深,动作更快](https://qwenlm.github.io/blog/qwen3/)
- [笑脸变形器- Qwen3](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3)
- [vLLM Qwen3](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py)
- [mlx -lm] Qwen3](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py)
- [Qwen3 技术报告](https://arxiv.org/abs/2505.09388)

{{#include copyright.md}}
