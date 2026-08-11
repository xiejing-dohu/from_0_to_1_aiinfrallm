# 🚧 第 3 周可选扩展：混合专家模型

>            本章正在审查中,可能有所改动。
在本章中,我们将实施向导的形状: **混合物
专家**,或 **MoE**,用于 Qwen3 家庭 家庭

此扩展是可选的 。 它改变模型的向导层 但不
排程器、调用缓存或注意合同,这样学生就可以完成
没有它,第3周的引擎。

目前为止,每个变压器的块 tiny-llm 已经使用同样的密度 Qwen3 MLP :

```plain
x -> gate_proj
x -> up_proj
SiLU(gate_proj(x)) * up_proj(x) -> down_proj
```

这是一个SwiGLU MLP。 每个 token 访问相同的重量。

MoE 仅改变变压器块中向导的一半。 而不是一个
稠密的MLP,模型拥有许多专家的MLP. 一个小路由器选择哪个专家
每个 token 应使用:

```plain
token hidden state -> router -> top-k experts -> weighted expert outputs
```

注意路径不变。 KV cache 不变。 人烟稀少
在MLP的半块。

**阅读**

- [超大神经网络:精密的合成层](https://arxiv.org/abs/1701.06538)
- [GShard:带有条件计算和自动硬化的放大巨型](https://arxiv.org/abs/2006.16668)
- [开关变换器:以简单高效的尺寸放大到千兆参数模型](https://arxiv.org/abs/2101.03961)

## Dense MLP 对 MoE MLP 语言

密度 Qwen3 第1周的MLP有一套重量:

```plain
w_gate: hidden_dim, dim
w_up:   hidden_dim, dim
w_down: dim, hidden_dim
```

Quen3-MoE小块有这些重量的银行:

```plain
expert_gate: num_experts, moe_hidden_dim, dim
expert_up:   num_experts, moe_hidden_dim, dim
expert_down: num_experts, dim, moe_hidden_dim
```

路由器每个专家产生一个分数:

```plain
router_logits: B, L, num_experts
router_probs:  softmax(router_logits)
```

那模特儿就选了 `num_experts_per_tok` 各专家 token:

```plain
expert_ids:    B, L, num_experts_per_tok
expert_scores: B, L, num_experts_per_tok
```

每个 token,只有那些选定的专家运行。 产出是加权的
总结如下:

```plain
output[token] = sum(score_i * expert_i(token))
```

这是中心 MoE 想法:模型可以包含许多参数,但每个参数
token 只激活其中的一个小子集。

## Quen3-MoE 形状

Quen3-MoE 保持与 Qwen3,包括QK规范,GQA,
RoPE,同样 KV cache 接口。 它将一些密集的 MLP 层替换为
人烟稀少 MoE 块。

有用的文件有:

- `gate`: 从隐藏大小到路由器线性层 `num_experts`
- `switch_mlp`: SwiGLU 多名专家 `moe_intermediate_size`
- `num_experts_per_tok`专家人数a token 用途
- `norm_topk_prob`: 选定专家分数是否重新正常化
- `decoder_sparse_step` 和 `mlp_only_layers`: 哪些层比较稀疏, 哪些层比较稠密

我们追踪的Quen3-MoE区没有共享专家。 人烟稀少
feed-forward输出只是加权的顶-k专家混合物.

## 分组定量 Matmul

MLX 不给我们一个高级 MoE 块中 `mlx.nn`它确实有一个
低级原始, `mx.gather_qmm`,执行定量矩阵
在为每行选择不同的矩阵时进行乘法。 在本章中,
我们将建立一个狭义的教学版本来阐述这一想法:
`grouped_quantized_matmul`.

为: MoE这意味着:

```plain
token rows:  N, D
expert ids:  N
weights:     E, O, D packed as 4-bit QuantizedWeights
output:      N, O
```

该行 `expert_ids[i] = e` 应乘以 `weights[e]`.

任务1将假设行已被专家ID排序 。 那个 MoE 帮助者将
从类型中保留反向顺序,使结果恢复到
原文 token 秩序。 秩序。

## 路由器步骤

路由器只是一个定量的线性层:

```python
router_logits = quantized_linear(x, w_router)
router_probs = softmax(router_logits, axis=-1)
```

对于一批代币:

```plain
x:             B, L, D
router_logits: B, L, E
router_probs:  B, L, E
```

地点 `E = num_experts`.

Quen3- MoE 然后使用顶级选择 :

```python
expert_ids = argpartition(-router_probs, k)[:k]
expert_scores = take_along_axis(router_probs, expert_ids)
```

若为 `norm_topk_prob` 是真实的, 重新正常化 `expert_scores` 因此所选的分数
每笔1美元 token.

## 专家步骤

每个专家都是同一类的SwiGLU MLP,我们已经知道:

```plain
expert(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

执行工作应创造象征性的专家工作,按专家分类,并开展
专家预测 `grouped_quantized_matmul`:

```plain
selected expert ids -> expanded token-expert rows
expanded rows -> sort/group by expert id
grouped expert rows -> grouped gate/up projection
SiLU(gate) * up -> grouped down projection
restore original token/top-k order -> weighted sum
```

重排是模式实施的一部分. 都留着了 token 该行
同一专家连成一体,以便应用分类矩阵
乘论.

## 任务1:分组量化 Matmul

```
src/extensions/src/quantized_matmul.cpp
src/extensions/src/quantized_matmul.metal
src/tiny_llm/quantize.py
src/tiny_llm/moe.py
```

执行 `grouped_quantized_matmul`,然后从 `grouped_expert_linear`.
这是组数- matmul 核心 MoE.

此可选界面是 **故意不预先申报** 输入
`src/extensions/src/tiny_llm_ext.h`,绑定,或核心CMake目标。
所需的 Week 2/3 接口由设置脚手架组成, 但此可选
章是已显示的:如果选择扩展变体,请添加新的
`tiny_llm_ext::grouped_quantized_matmul` 声明,具有约束力, C++ 来源
函数, `grouped_quantized_matmul` Metal 内核,在这里建立注册。
然后修改已有的 `grouped_expert_linear` 函数在
`src/tiny_llm/moe.py` 来称呼它。 让它远离核心启动器
从早期看来需要的可选的未来接口
检查站。

`grouped_quantized_matmul` 接受:

```plain
a:           R, D
w_experts:   packed QuantizedWeights for num_experts, output_dim, D
expert_ids:  R, sorted by expert id
```

它返回:

```plain
out:         R, output_dim
```

每行使用匹配行所选的专家 `expert_ids`:

```plain
out[row] = a[row] @ dequantize(w_experts[expert_ids[row]]).T
```

实施工作应:

```plain
1. add a Python wrapper for grouped_quantized_matmul,
2. extend the quantized matmul extension with a grouped entrypoint,
3. read expert_ids[row] inside the kernel,
4. use that expert id to choose the expert weight, scale, and bias row.
```

之后,执行 `grouped_expert_linear` 输入 `src/tiny_llm/moe.py`:

```plain
1. flatten token rows and expert ids,
2. sort rows by expert id,
3. call grouped_quantized_matmul,
4. restore the original order.
```

电话应该看起来像:

```python
out = grouped_quantized_matmul(
    w_experts.scales,
    w_experts.biases,
    group_size=w_experts.group_size,
    bits=w_experts.bits,
    a=grouped_rows,
    b=w_experts.weight,
    expert_ids=grouped_expert_ids,
    transpose_b=True,
)
```

此任务映射到与 `QuantizedSwitchLinear` 输入 `mlx-lm`: 每个
token 行使用不同包装的专家矩阵,专家ID选择
右矩阵。

## 任务2:路由器 Top-k

```
src/tiny_llm/moe.py
```

修改现有的 `route_topk` 函数在此文件中。

执行 `route_topk`。它接受隐藏状态和路由器重量,然后
返回:

- 路由器概率
- 选定的专家身份
- 选定的专家得分

使用 `quantized_linear` 和 `softmax`启用 `mx.argpartition` 选择顶部
`num_experts_per_tok` 那么,专家 `mx.take_along_axis` 收集他们的分数。

保留 `norm_topk_prob` 由于 Quen3- MoE 将这种行为存储在
模型配置。

## 任务3: Qwen3 分页 MoE 块

```
src/tiny_llm/moe.py
```

修改 `Moe.__init__` 和 `Moe.__call__`,编译
`grouped_expert_linear` 和 `route_topk` 任务 1-2。

执行 `Moe` 通过组成任务1和任务2:

```plain
hidden states -> route_topk
hidden states + expert ids -> grouped gate projection
hidden states + expert ids -> grouped up projection
SiLU(gate) * up -> grouped down projection
weighted sum over num_experts_per_tok
```

这完成了Qwen3-MoE稀疏的饲料前进块. 没有共同的专家
这个街区的分支。

## 任务4:整合 Qune3- MoE 层

```
src/tiny_llm/qwen3_week3.py
src/tiny_llm/models.py
```

修改 `is_qwen3_moe_sparse_layer` 和 `Qwen3ModelWeek3.__init__` 输入
`src/tiny_llm/qwen3_week3.py`外加时 `dispatch_model` 输入
`src/tiny_llm/models.py`.

添加重用第3周的 Quen3- MoE 加载器路径 Qwen3 注意和呼呼 KV
缓存行为,但将选定的块 MLP 互换为 `Moe`.

模型包装器应:

- 留着 Qwen3 注意不变,
- 定期使用 `Qwen3MLP` (单位:千美元) `mlp_only_layers`,
- 使用 `Moe` 用于选择的稀疏层
  `decoder_sparse_step`,
- 载运路由器和专家重量 `QuantizedWeights` 从 Quen3- MoE 中 MLX
  型号,
- 保持不变 decode 调用形状 :

```python
logits = model(tokens, offset, cache)
```

无调度器 API 变动 `src/tiny_llm/batch.py` 正确性是必需的。

使用:

```bash
pdm run test --week 3 --day 6
```

通过正常的生成输入点运行此任务, 而不是添加一个
分开服务入口。 例如:

```bash
hf download Qwen/Qwen3-30B-A3B-MLX-4bit

pdm run main --solution tiny_llm --loader week3 --model qwen3-30b-a3b \
  --prompt "Give me a short introduction to mixture of experts."

pdm run batch-main --solution tiny_llm --loader week3 --model qwen3-30b-a3b \
  --batch-size 2 --prefill-step 16
```

{{#include copyright.md}}
