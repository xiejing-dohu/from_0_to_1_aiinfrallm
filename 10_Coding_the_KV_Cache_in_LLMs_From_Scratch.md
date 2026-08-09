# 10. Understanding and Coding the KV Cache in LLMs from Scratch（从零手写实现 KV Cache 机制）

> **导读**：本章节精读并深度解析了 AI 领域知名专家 Sebastian Raschka 博士（《Build a Large Language Model From Scratch》作者）的经典技术讲义 *Understanding and Coding the KV Cache in LLMs from Scratch*。内容涵盖 KV Cache 的直观原理、PyTorch 代码从零手写实现、无 Cache vs 有 Cache 性能推导对比，以及生产环境下预分配显存（Pre-allocation）与滑动窗口（Sliding Window）等进阶优化。
> 
> * **原文链接**：[Understanding and Coding the KV Cache in LLMs from Scratch (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)
> * **开源代码参考**：[rasbt/LLMs-from-scratch (GitHub)](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/03_kv-cache)

---

## 1. 为什么需要 KV Cache？（痛点与核心直观概念）

在 LLM 自回归生成（Autoregressive Generation）文本时，模型是**逐字（Token-by-Token）**生成后续内容的。

### 1.1 无 KV Cache 时的冗余计算
假设 Prompt 是 `"Time"`，模型依次生成 `"flies"` 和 `"fast"`：
* **Step 1**：输入 `"Time"`，计算输入序列的注意力，输出 Token `"flies"`。
* **Step 2**：输入 `"Time flies"`，**重新对全量序列 `"Time flies"` 从头进行 Embedding 和 Attention 计算**，输出 Token `"fast"`。
* **Step 3**：输入 `"Time flies fast"`，再次重新编码整个前缀...

```
Step 1:  [Time] ---------------> 生成 "flies"
Step 2:  [Time] [flies] --------> 生成 "fast"  (冗余计算了 "Time" 的 K, V)
Step 3:  [Time] [flies] [fast] -> 生成 "next"  (冗余计算了 "Time", "flies" 的 K, V)
```

可以看到，随着生成序列长度 $n$ 增加：
* **无 KV Cache 计算复杂度**：每一步都需要重新计算前 $t$ 个 Token 的 Key 和 Value 矩阵，累计计算量随序列长度呈**二次方增长 $O(n^2)$**。
* **核心浪费**：前 $t-1$ 个 Token 的 Key（$K$）和 Value（$V$）张量在自回归过程中是固定的，完全没有必要在每个 Step 重新做一次矩阵乘法！

---

## 2. KV Cache 工作原理

在 Transformer 的 Self-Attention 模块中：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

KV Cache 的思想极其直观：
1. **Prefill（预填阶段）**：一次性处理 Prompt 中的所有 Token，计算它们的 $Q, K, V$。把计算出的 $K$ 和 $V$ 保存在缓存中（`cache_k`, `cache_v`）。
2. **Decode（解码阶段）**：每次只传入**最新生成的 1 个 Token**。
   * 只计算该新 Token 的 $Q_{new}$, $K_{new}$, $V_{new}$；
   * 将 $K_{new}$ 与 $V_{new}$ 追加（Concatenate）到先前保留的 `cache_k` 与 `cache_v` 尾部；
   * 用 $Q_{new}$ 与完整的 `cache_k`（包含历史与最新 Token 的 Key）做点积 Attention，生成当前 Step 的 Output。
3. **复杂度下降**：将每一步的计算量从 $O(t)$ 降到了 $O(1)$，总体累计计算复杂度从 $O(n^2)$ 降到了 **线性 $O(n)$**。

---

## 3. 从零手写实现 KV Cache（PyTorch 代码剖析）

### 3.1 注意力机制层实现（Self-Attention with KV Cache）

下面是带有 KV Cache 逻辑的 Causal Self-Attention 模块代码逻辑：

```python
import torch
import torch.nn as nn

class MultiHeadAttentionWithKVCache(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out)

        # 初始化 KV Cache 容器
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def reset_kv_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape

        # 1. 计算当前输入的 Q, K, V
        keys = self.W_key(x)     # (b, num_tokens, d_out)
        values = self.W_value(x) # (b, num_tokens, d_out)
        queries = self.W_query(x)# (b, num_tokens, d_out)

        # 重塑张量格式以支持 Multi-Head: (b, num_heads, num_tokens, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. KV Cache 拼接逻辑
        if use_cache:
            if self.cache_k is None:
                # Prefill 阶段：缓存整个 Prompt 的 K 和 V
                self.cache_k = keys
                self.cache_v = values
            else:
                # Decode 阶段：将新 Token 的 K, V 追加到已有 Cache 尾部
                self.cache_k = torch.cat([self.cache_k, keys], dim=2)
                self.cache_v = torch.cat([self.cache_v, values], dim=2)

            keys = self.cache_k
            values = self.cache_v

        # 3. 计算 Scaled Dot-Product Attention
        # queries shape: (b, num_heads, q_len, head_dim)
        # keys shape:    (b, num_heads, kv_len, head_dim)
        attn_scores = queries @ keys.transpose(-2, -1) / (self.head_dim ** 0.5)

        # 如果未开启 Cache，应用标准的下三角 Causal Mask
        if not use_cache or queries.shape[2] > 1:
            mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(mask, -float("inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, -1)
        return self.out_proj(context)
```

---

### 3.2 文本生成调度函数对比

在生成循环中，开启 `use_cache=True` 后只需要给模型送入最后一个新生成的 Token：

```python
def generate_text_with_kv_cache(model, idx, max_new_tokens, ctx_len):
    model.reset_kv_cache()
    
    # 1. Prefill 阶段：传入完整 Prompt 初始化 KV Cache
    with torch.no_grad():
        logits = model(idx[:, -ctx_len:], use_cache=True)
    
    for _ in range(max_new_tokens):
        # 取最新位置 logits 采样出下一个 token
        next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_idx], dim=1)
        
        # 2. Decode 阶段：每次仅向模型送入单字 next_idx！
        with torch.no_grad():
            logits = model(next_idx, use_cache=True)
            
    return idx
```

---

## 4. 性能对比与工程调优

在 Apple M4 芯片（124M GPT-2 模型生成 200 Token）的基准测试中：
* **无 KV Cache**：耗时约 **1.55 秒**
* **基础 KV Cache (`torch.cat`)**：耗时约 **0.31 秒**（实现 **~5 倍加速**！）

且两者输出的 Logits 和最终生成的文本概率**完全一致（Zero Discrepancy）**，验证了索引对齐的正确性。

---

## 5. 生产环境进阶优化避坑

在真实生产推理框架（如 vLLM / TensorRT-LLM）中，上述简单的 `torch.cat` 实现存在两个严重缺陷：

### 5.1 痛点一：频繁 `torch.cat` 引发显存碎片
* **原因**：每次追加新 Token 都要重新申请一块更大的连续 GPU 显存并做数据拷贝，导致极其严重的显存碎片化与 Kernel 启动开销。
* **解决方案 1（预分配显存 Pre-allocation）**：
  在推理开始前，根据 `max_seq_len` 直接分配好固定形状的空 Tensor，每次填充切片：
  ```python
  cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
  cache_k[:, :, ptr:ptr+1, :] = new_k  # 仅做切片赋值，零内存重分配
  ```

### 5.2 痛点二：显存无界线性膨胀
* **原因**：当上下文达到几十万 Token 时，KV Cache 体积会轻松撑爆 GPU 显存。
* **解决方案 2（滑动窗口截断 Sliding Window）**：
  仅保留最近 `window_size` 个 Token 的 KV：
  ```python
  cache_k = cache_k[:, :, -window_size:, :]
  cache_v = cache_v[:, :, -window_size:, :]
  ```
* **解决方案 3（分页管理 PagedAttention）**：
  如第 1 章与第 4 章所述，将 KV Cache 拆分为固定 Block（如 16 Tokens/Block），通过操作系统级别的逻辑页表管理，实现物理显存的高效复用。

---

## 6. 优缺点对比总结

| 维度 | **无 KV Cache** | **有 KV Cache** |
| :--- | :--- | :--- |
| **计算复杂度** | $O(n^2)$（二次方暴增，随序列变长极慢） | $O(n)$（线性复杂度，Decode 每步仅 $O(1)$） |
| **显存占用** | 低（仅保留 Activation 中间值） | 高（必须长期在 GPU 显存保留 KV 矩阵） |
| **适用于** | 模型训练阶段（Training） | 模型在线推理阶段（Inference） |
| **代码复杂度** | 极简 | 较复杂（需维护 Cache 索引与动态 Mask） |
