# 🚧 第 2 周第 1 天：KV Cache

> **现状:实验。** 见
> [第2周核查矩阵](./week2-overview.md#verification-status) (单位:千美元)
> 不断测试、在当地测量和仍在审查中的内容。

在本章中,我们将增加: **密钥值缓存** 页:1 Qwen3 型号。 期间
生成时,缓存允许每个关注层重用来自
上一个符号,而不是每一步重算整个前缀。

这是第二周的基础 decode 优化, 不只服务 3周
特性。 没有它,每一个生成 token 重运行所有模式层
不断增长的前缀,压倒了更快的单个内核带来的收益.

**读数**

- [KV缓存解释:优化变形器推理效率](https://huggingface.co/blog/not-lain/kv-caching)

回顾第1周如何反复向模型提供完整的序列:

```plain
tokenized_prompt: [1, 2, 3, 4, 5, 6]
prefill: _step(model, [1, 2, 3, 4, 5, 6]) # returns 7
decode:  _step(model, [1, 2, 3, 4, 5, 6, 7]) # returns 8
decode:  _step(model, [1, 2, 3, 4, 5, 6, 7, 8]) # returns 9
...
```

```plain
x: B, L, E
q = linear(x, wq) -> B, L, H_q, D
k = linear(x, wk) -> B, L, H, D
v = linear(x, wv) -> B, L, H, D
q = rms_norm(q, q_norm)
k = rms_norm(k, k_norm)
q = rope(q, offset=slice(offset, offset + L))
k = rope(k, offset=slice(offset, offset + L))
(transpose as needed)
x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, L, H_q, D
# q/k/v and the returned model tensor are BF16; the Python `mlx.core` expression may use FP32 intermediates
(transpose as needed)
x = linear(x, wo) -> B, L, E
```

注意机制计算如下:

$$
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$


考虑两个连续的解码步骤 `L = S = 3` 和 `L = S = 4`.
假设每个关注头都有维度 `D = 4`:

```
L = 3
Q        x  K^T     =         
1 1 1 1     1 2 3      1x1  -inf -inf
2 2 2 2     1 2 3      2x1  2x2  -inf
3 3 3 3     1 2 3      3x1  3x2  3x3
            1 2 3

L = 4
Q        x  K^T       =
1 1 1 1     1 2 3 4      1x1  -inf -inf -inf
2 2 2 2     1 2 3 4      2x1  2x2  -inf -inf
3 3 3 3     1 2 3 4      3x1  3x2  3x3  -inf
4 4 4 4     1 2 3 4      4x1  4x2  4x3  4x4
```

领先者 `3 x 3` 块 `QK^T` 两个步骤都是一样的。 一个因果面具
也使先前的查询无法处理新的 token因此,他们的产出
不变。 重新计算这些行,他们的 softmax 价值及其产品
与 `V` 是浪费的工作。 只有新查询行提供新输出 。

相反,缓存以前的密钥和值,并只计算预测值
来信符 :

```
K in cache:
1 1 1 1
2 2 2 2

[a b c d] represent cached values

L = 1, S = 3
Q        x  K^T       =         
            (⬇️ is K not transposed)
            [1 1 1 1]      
            [2 2 2 2]      
3 3 3 3      3 3 3 3      3x1 3x2 3x3

L = 1, S = 4
Q        x  K^T       = 
            (⬇️ is K not transposed)
            [1 1 1 1]      
            [2 2 2 2]      
            [3 3 3 3]
4 4 4 4      4 4 4 4      4x1 4x2 4x3 4x4
```

## 任务 1: 执行键值缓存

```
src/tiny_llm/kv_cache.py
```

每个变形器层都维持自己的键值缓存. 缓存显示一个
方法, `update_and_fetch`,其中:

1. 接受新计算 `K` 和 `V` 给即将到来的标志
2. 沿序列维度附加它们。
3. 返回全部缓存 `K` 和 `V`,更新的偏移,和面具。

在本章中,缓存通过 `mask` 改为不使用
`mask_length`在第三周中,这些参数对分批处理很重要。

您可以在 `kv_cache.py` 作为 `TinyKvFullCache`:

```plain
L_new = number of incoming tokens

update_and_fetch(key, value, mask_length, mask) -> key, value, offset, mask

key:   B, H, L_new, D
value: B, H, L_new, D

if self.key_values is None:
    self.key_values = (key, value)
else:
    cached_key, cached_value = self.key_values
    self.key_values = (
        concat(cached_key, key, axis=2),
        concat(cached_value, value, axis=2),
    )

self.offset += L_new
key, value = self.key_values  # B, H, offset, D

return key, value, self.offset, mask
```

这是故意的 简单的密集基线,不是生产 KV cache.
`mx.concat` 分配更大的缓冲器并复制以前的 K/V 内容
每一个成长步骤。 {\fn黑体\fs22\bord1\shad0\3aHBE\4aH00\fscx67\fscy66\2cHFFFFFF\3cH808080}一个逐字标语 decode 长度 `S`,这些副本添加
上限 `O(S²)` 字节,即使缓存避免 `O(S²)` 前缀重算。
引用缓存记录此流量为 `growth_copy_bytes` 所以描述器
可以把它和注意力分开 第3周将基准替换为
预分配页面; 不将重复的连接设计复制到 a
服务缓存。

## 任务2:构建缓存周2模式

```
src/tiny_llm/qwen3_week2.py
```

保留第一周 Python 模型及其全前缀生成循环不变。
单独开始 `qwen3_week2.py` 具有相同密度重量的模型
第1个星期 `mlx.core` RMSNorm, RoPESwiGLU,还有注意方程式 只更改状态流
本章:第2周模式接受缓存和抵消,而第1周保留
重新计算完整的前缀. 这就得出了基线,即以后每两周
章节将优化。

- 给每一层自己的缓存。
- 加一个 `offset` 对模型的参数。 这是已经输入的符号数
  缓存,因此,第一个进入者的位置 token.
- 参数应该符合缓存当前序列长度。 请求可以
  说清楚一点
- 呼叫器和缓存器都跟踪偏移 以方便一致性检查。

计算流程示例 :

```plain
x: B, L, E
q = linear(x, wq) -> B, L, H_q, D
k = linear(x, wk) -> B, L, H, D
v = linear(x, wv) -> B, L, H, D
q = rms_norm(q, q_norm)
k = rms_norm(k, k_norm)
q = rope(q, offset=slice(offset, offset + L))
k = rope(k, offset=slice(offset, offset + L))
transpose q, k, v to B, H, L, D
k, v = cache.update_and_fetch(k, v)  # k/v: B, H, S, D; q: B, H_q, L, D
x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, H_q, L, D
# q/k/v and the returned model tensor are BF16; attention arithmetic is still the Week 1 `mlx.core` path
transpose and reshape x to B, L, H_q * D
x = linear(x, wo) -> B, L, E
```

在这里, `L` 是进入的查询符号和 `S` 是总缓存
更新后的序列长度。 这与第一次全球质量评估周的公约相符: `L`
是查询长度,而 `S` 是键/值源长度。 期间
单脚解码, `L = 1` 和 `S` 每个电话都长一个

线性层, RMSNorm, RoPESwiGLU, 关注仍然是第1周
Python 该检查站的执行情况。 不要引入包装重量或快速
内核还原:测量一个算法变化使收益归属.
该模型仍然使用 BF16 存储; "Week 1 Python" 描述
执行样式,而不是返回 FP32 型号。

## 任务 3: 创建请求扫描卡片

```
src/tiny_llm/qwen3_week2.py
```

执行 `create_kv_cache` 所以每个请求都有一个缓存手柄
变形层. 将匹配层缓存通过每个块并保存
与缓存的逻辑长度一致的调用者抵消。

为了验证正确性,进行以下测试,这类似于第1周
模型测试 :

```bash
pdm run test --week 2 --day 1
```

## 任务4:连接服务循环

```
src/tiny_llm/generate.py
```

第一个型号调用预补缓存,并带有完整的快捷键. 以后的每一个
电话只通过 token 前一个步骤编制的文件
已缓存的符号数。 同样的生命周期将归
第3周的连续计时器

例如:

```plain
tokenized_prompt: [1, 2, 3, 4, 5, 6]
prefill: _step(model, [1, 2, 3, 4, 5, 6], 0)  # returns 7
decode:  _step(model, [7], 6)  # returns 8
decode:  _step(model, [8], 7)  # returns 9
...
```

您可以测试您的解决方案 :

```bash
pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-4b
```

您也可以使用参考解决方案运行相同的循环 :

```bash
pdm run main --solution tiny_llm_ref --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-4b
```

## 综合和计量

在更改任何运算符前运行缓存的一周 1 检查点结束 :

```bash
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-4b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2
```

将此数字记录在您的优化分类账中 。 接下来的一章教如何
与第1周和 MLX; 以后的每个命令都完全更改
一个累计检查站。

第1天是一个算法检查点,所以它不会发明阴影层
从 a 限制 GPU 追踪。 检查站取消完全前置重置; 2.
使用端到端基准来衡量算法变化。 第2天措施
并识别投影-重量带宽瓶颈。 第 3 条
引入 4- 位量化并执行 SIMD 内核
直接用包装重量操作。

{{#include copyright.md}}
