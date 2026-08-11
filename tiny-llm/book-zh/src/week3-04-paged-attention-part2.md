# 🚧 第 3 周第 4 天：直接分页注意力

>            本章正在审查中,可能有所改动。

在本章中,我们将建立 **直接呼叫注意**。排程器通过
请求本地块表格和上下文长度到 a GPU 内核,读取 K/V
从共享的地层池 而不首先收集密集的批量。

> **先决条件:** 完成第3周 第3天的页面存储和第2周 第5天的页面存储
>在线 softmax关注. 这里的新概念是翻译逻辑 K/ V
> 通过块表的位置。 铺设 FlashAttention 之后才来
>直接路径工作.

## 页 次 KV Cache vs 页面注意

这两个想法是相互关联的,但它们并不相同:

1. **页 次 KV cache**
   KV以固定大小的页面存储.
2. **请注意**
   注意路径通过页面表等元数据直接从这些页面读取KV.

你可以执行第一个,没有第二个, 但真正的服务支付 当两者都在场。

## 页面运行时需要的元数据

KV 被调用后, 密度 `B x H x S x D` 升降机不再是自然的运行时间代表。 相反,运行时间应编制如下元数据:

```plain
block_table:  [B, max_pages_per_request]
context_lens: [B]
```

对于当前执行的图层 :

- `block_table[b, i]` 提供请求的页码 `b`当前层次的逻辑页面 `i`
- `context_lens[b]` 给出有效的 token 请求数 `b`

这是调度器和注意力内核之间的桥梁.

制作运行时间往往也带有写侧元数据,例如: `slot_mapping`.
对于这一章,我们把写面放在缓存内,并专注于读面
需要注意的元数据。

## 为什么 `block_table` 事项

假设请求A的一层缓存有:

```plain
page_ids = [12, 5, 3]
context_len = 10
page_size = 4
```

然后逻辑序列位置映射到物理存储中像这样:

```plain
logical 0..3  -> page 12
logical 4..7  -> page 5
logical 8..9  -> page 3
```

注意时间不需要完全聚集的密集的呼声,如果它已经知道:

- 每一逻辑块的当前页,
- 环境有多长,
- 和当前查询位置的位置。

就是这个 `block_table` 和 `context_lens` 编码。

## 页面注意 API

目前,运行时间应增加一个新的关注点:

```python
paged_attention(
    query,
    key_pages,
    value_pages,
    block_table,
    context_lens,
    page_size,
    scale=None,
    mask="causal",
)
```

形状如下:

```plain
query:          B, H_q, L, D
key_pages[i]:   1, H_kv, page_size, D
value_pages[i]: 1, H_kv, page_size, D
block_table:    B, max_pages
context_lens:   B
```

源长度不再以一个毗连的拉伸维度表示.
操作员从页面表格中逻辑地重建它.

在本章中, `paged_attention` 改为: GPU
内核. 运行时间合同现在是:模式代码和批量代码通过页
加上元数据, 和关注内核走的元数据没有第一个
重建稠密的 K/ V.

## Prefill 元数据

期间 prefill,块可能跨越多个页。 运行时间需要知道:

- 已经存在的当前层次的页面,
- 分配了哪些新网页,
- 尾页上有多少有效标志
- 如何将收到的K/V行映射到页面存储中。

在这个教学实施中,缓存仍然拥有写侧记账.
注意路径只需要写完后块表即可.

## Decode 元数据

期间 decode,每个活动请求通常写一个 token.

运行时间应能够:

1. 附录 tokenK/V 到当前尾页,
2. 只在尾页满时分配新页,
3. 更新当前图层缓存 `context_len`,
4. 利用 `block_table`

这是重点 decode 从第1天起停止支付重复的密集重包装成本.

## 选择每个查询形状的日程

执行之前 GPU 路径, 单独 decode 从 prefill单片
形状无法保存 GPU 忙于一个单键查询和一个长的提示。
使用这些设计规则:

1. 保留第二周 BF16 模型边界和再利用内部积累
   政策不变。
2. 对于简短的询问,要披露缓存背景的平行性。 别
   为不存在的查询行保留大部分线索组。
3. 用于 prefill,从一个直接的页面行走时间表开始,其地址
   计算容易验证。
4. 使用可读方程式 `mlx.core` 和密集的第二周
   注意你解决方案中的内核 作为新事物的正确性预言
   页面行走时间表。

从此调度计划开始, 并将其阈值作为值来验证
你的硬件 :

| 形状 | 在您的解决方案中发送 | 工作分解 |
|---|---|---|
| `L <= 8` | 矢量页 decode | 每个查询行一个线程组; 32 SIMD 组合超越上下文并部分合并 `(max, sum, output)` 各州。 |
| `L > 8` | 直接页 prefill | 将逻辑 K/V 瓦片通过块表,并保持时刻表可故意检查. 第5天优化. |

将形状决定置于扩展边界, 而不是转换输入
或回到密集的注意力在 Python立即确定基准值
低于或高于每个阈值,同时保持模型化
`paged_attention` API 不变。

## 此地图将如何 `tiny-llm`

### `src/tiny_llm/attention.py`

添加新函数 :

```python
def paged_attention(...):
    ...
```

在你的解决方案中,让它正确 第一页行走
Metal 内核为 online softmax:

1. 使用 `block_table[b]` 查找请求的物理页面 `b`,
2. 使用 `context_lens[b]` 忽略未使用的尾部容量,
3. 用小瓷砖访问K/V,而不是实现密集的K/V,
4. 将每个瓦片合并到输出中 online softmax.

从密集关注中的重要变化是K/V地址计算. 请注意
通过密集的 K/V 通过指针算术推进。 第3周必须各翻译
通过逻辑密钥位置 `block_table` 第一个:

```plain
logical key position -> logical page -> physical page id -> slot in page
```

查完后,在线软max更新与第2周相同
4号 保持页面行走时间表的简单 使块表和
尾页边界错误是可见的。 第5天将铺平内部矩阵工作
同时保留此地址计算。

单键 decode 需要不同的工作分解。 六十四号 prefill 瓷砖
几乎每个查询行都会闲置, 所以发送简短查询到一个
面向矢量的内核,将上下文分割开来 SIMD 团体和团体
合并其部分在线软件状态. 不运行 decode 通过固定
三十二行斜线 prefill 瓷砖。

因此,页面库应暴露毗连的物理存储:

```plain
key_pages:   P, H_kv, page_size, D
value_pages: P, H_kv, page_size, D
```

A Python 页面浏览器列表对教学分配器是方便的,但是
GPU 内核需要一个单一的缓冲 `page_id` 可以变成地址。

### `src/tiny_llm/qwen3_week3.py`

注意模块应直接调用调用时间:

```python
metadata = cache.update_and_fetch_paged(...)
x = paged_attention(...)
```

第3周缓存手柄预计将提供页面元数据. 如果密密的缓存是
传递到第3周的模型,这是一个编程错误,而不是一个信号
悄悄地回到密集的注意力中.

### `src/tiny_llm/batch.py`

调度器现在需要编写运行时元数据, 而不是只编写密集的 K/ V :

- 每个活动请求的每层页表
- 添加批号 `block_table`
- `context_lens`

这是连续的批量和呼号注意终于连接的地方. 第1天的分批工作是通过重新包装收发机。 在此,分批工作应该通过重新使用页面表格和仅更新新的插槽来进行。

## 执行令

使用此执行命令 :

1. 网页存储
2. `block_table` / `context_lens` 管道
3. 正确性-第一页行走 GPU 请注意
4. 型号和批量发送

每个步骤在添加下一个抽象之前都有一个直接的正确性检查.

## 是什么必须持有,什么打破 如果它不

这些是值得检查的变种:

1. **`context_len` 等于写逻辑数 token 职位。** 如果是这样
   过于小, 注意跳过写入 K/ V; 如果太大, 注意
   读取未写尾槽。 两种情况都使页码输出与
   密集基线。
2. **`block_table` 重建与密度相同的逻辑 K/ V 顺序
   基线。** 错误的映射可以将查询与错误对齐 token其K/V和
   即使每页都包含有效的数据,也要修改输出。 重排顺序
   完整页面可以更改一个因果前缀结果,因为它会改变
   K/V对每个查询可以看到. 反之,单脚 decode 总之
   完整页中的位置按顺序变化
   每个K/V对一起移动。
3. **分配器只给每个页面一个活的缓存手柄, 除非共享
   很明显。** 如果两个活柄别名一个页面, 写一个请求
   覆盖其它请求仍然可以处理的 K/ V 。
4. **重放请求返回每层缓存所拥有的每一页
   曾经** 缺少一页漏水池容量;返回一次提高
   池已经没有错误, 而不是完成清理 。
5. **Decode 仅在尾页溢出时分配新页面。** 分配
   较早的线条可写尾槽,并夸大旧页数。 这个
   课程库增加而不是报告耗尽,这样废物就可以强迫
   支持存储以增长和复制更早,增加内存压力.

## 任务 1: 添加批次元数据

```
src/tiny_llm/paged_kv_cache.py
src/tiny_llm/kv_cache.py
src/tiny_llm/batch.py
```

修改 `TinyKvPagedCache.block_table`, `context_lens`, `paged_metadata`,以及
`update_and_fetch_paged` 输入 `src/tiny_llm/paged_kv_cache.py`。然后更新
`Request.try_prefill` 和 `_step` 输入 `src/tiny_llm/batch.py` 携带这些
每个活动请求的数组。

扩展批量缓存和排程器,以便准备:

- `block_table`
- `context_lens`

对所有积极的请求。

## 任务2:定义 `paged_attention`

```
src/tiny_llm/attention.py
src/extensions/src/paged_attention.cpp
src/extensions/src/paged_attention.metal
```

修改这些精确启动函数 :

- `paged_attention` 输入 `src/tiny_llm/attention.py`;
- `tiny_llm_ext::paged_attention`, `PagedAttention::eval_cpu`,以及
  `PagedAttention::eval_gpu` 输入 `src/extensions/src/paged_attention.cpp`;
- `paged_attention_decode` 和 `paged_attention_scalar_f32` 输入
  `src/extensions/src/paged_attention.metal`.

这个检查点也将已经可读的量子化 token 查询
第3周的单发路径。 修改 `QuantizedEmbedding.__call__` 输入
`src/tiny_llm/embedding.py`, `tiny_llm_ext::quantized_embedding` 加号
`QuantizedEmbedding::eval_cpu`/`eval_gpu` 输入
`src/extensions/src/quantized_matmul.cpp`,以及
`quantized_embedding_w4a16_g128` 输入
`src/extensions/src/quantized_matmul.metal`启动者声明,
已经存在两种操作的装订、支架和建造注册,
继续关闭,直到替换它们。

添加一个页面关注界面,其输入来自页面运行时间
比密集的重建 `S` 维度。 保留第二周的精度
不添加新模式的合同 dtype 或转换在服务层。

在保持在线软件状态的同时,

```python
running_max = max(previous_max, page_max)
running_sum = previous_sum * exp(previous_max - running_max) + page_sum
output = previous_output * exp(previous_max - running_max) + page_output
```

在所有可见的页面被耗尽后, 分隔 `output` 由 `running_sum`.
这是关键的想法,让内核避免实现稠密的K/V同时
仍然产生与密集关注相同的结果.

执行两个正确的第一 GPU 发送时间 :

1. 用于 `L <= 8`,分区逻辑上下文位置 SIMD 团体和团体
   合并其部分 `(max, sum, output)` 状态。 那个
   初始时间表使用 32 SIMD 每个查询的组。 解析物理页面
   一次,然后组 `g` 访问时段 `g`, `g + 32`, `g + 64`,例如:
   在该页内; 不分割并重新装入 `block_table` 百分比 token.
2. 对于较长的查询,在直接的页面行走时间表中指定查询行和
   解析每个 K/V 牌 `block_table`。当一个瓦片对齐时,
   无法跨越页面边界, 共享其一个物理页面 ID
   整个瓷砖。 对最后瓷砖业绩的有利所有权
   时间表。

将小的确定性固定装置与可读方程式进行比较
`mlx.core` 和密集的第二周 注意路径前调整页面行走
时间表。

最终 Qwen decode 时刻表,专业 BF16 `D = 128`: 每个车道拥有
Q,K,V四个毗连维度,以及输出. 毕竟所有上下文
访问职位,将32个部分输出矢量通过一个转换
紧凑32×32线组瓦. 每个 SIMD 然后减少四个维度
与 `simd_sum`将刮伤减少4.25KiB
每个卡片输出线程存储一个完整的部分矢量。 保留一个通用
BF16 用于其他头维的专业化,所以无法优化
悄悄地重新解释 `D = 32` 作为 `D = 128`.

### 您的解决方案的边界

MLX 仍为形状、重塑、转换、毗连的阵列运行时间
存储, dtype 转换、分配和自定义的原始发送。 那个
执行本身必须留在你的解决方案中:不要呼吁
`mx.fast.scaled_dot_product_attention`,再使用一个 MLX 注意/内核,或
重建密密的 K/ V , 并用 MLX matmul 加号
softmax. MLX SDPA只能作为外部在测试和基准中出现
正确性和性能基线。

两者 prefill 和 decode 通过此接口读取页面存储。 不添加一个
密度唯一特例:第5天优化了这个同页合同.

## 任务3:从模型发送

```
src/tiny_llm/qwen3_week3.py
```

修改 `Qwen3MultiHeadAttention.__call__`, `Qwen3ModelWeek3.__init__`,以及
`Qwen3ModelWeek3.__call__` 选择页面路径并启用自定义
仅嵌入此累积检查站 。

更新模型,以便当缓存提供分页运行的元数据时,它可以路径呼叫注意.

将 K/V 添加到页面库,并将其元数据传递给每个查询的注意
形状。 较长的查询使用本章直接的页式预补时间表;
简短的查询使用矢量页解码调度。 不改变路径缓存
dtype 或收集密集的 K/V 声波。

这创造了第4天的路线政策:

```plain
prefill or long chunk -> direct page-walking attention
decode or short chunk -> paged vector attention
```

第5天用页替换长排程 FlashAttention 无
改变这种模式政策。 `--disable-paged-attention` 属于第4天
密集的gather教学衰竭,而不是完成的服务路径.

## 任务 4: 连接到连续断层

```
src/tiny_llm/batch.py
```

修改 `Request.try_prefill`, `Request.decode_done`, `_step`,以及
`batch_generate`。清理请求必须呼叫 `TinyKvPagedCache.release` (单位:千美元)
每层缓存。

更新请求录入,槽重用,并请求删除,以便:

- 完成请求,免费他们的页面,
- 在这个教学实施中,这意味着从每一层缓存中释放页面,
- 从相应的层池分配新请求,
- 活动 decode 步骤重用页面元数据,而不是重建稠密的 K/V。

在本章之后,服务堆栈对真正的高通量运行时间有正确的结构: page不再仅仅是存储的诡计,而是执行模式本身的一部分.

## 测量直接页面行走

这个实验室的目标是决定何时直接翻页是有用的. 页 次
注意不是自动的更快的注意操作:它经常交易,
为灵活分配而连接的 K/V 访问,并取消密集的重新包装
否则会在人们注意之前发生。 您的测量必须包括两个
交易的两边

在同一台机器上记录三个操作员基线:

1. 解决方案中密集的周二关注路径,包括任何必要的
   K/V 组装,
2. 您的直呼路径,
3. MLX 注意力路径，作为生产级库的基线。

使用相同的 Quen3-4B decode 所有三条路径的形状。 密集控制必须
包括所需的页面到信息集; 直接路径读取同一页
元数据; MLX 行测量其已安装的注意操作器
集合声纳 :

```bash
pdm run bench-week3-attention --offline --contexts 128 1024 \
  --page-size 128 --warmup 5 --iterations 60 --repeats 4 \
  --cooldown-seconds 1 \
  --json-output benchmark_results/m4-pro-qwen3-4b-week3-attention-mlx-0.32.0.json
```

每个值为四个平衡的新工艺中位数的中位数,60
每个过程5次热身后同步呼叫 :

| 背景情况 | 强度 + 集合 | 直接页 | MLX 已安装 |
|---:|---:|---:|---:|
| 128 | 我们184.01 | 我们187.55 | 我们 |
| 1,024 | 我们420.88 | 我们249.79 | 我们207.18 |

直接转速比密度加热慢1.9%,128个指数,但40.7%
更快的1 024个令牌。 MLX 两种形状都保持更快 检查 BF16
输出与文件 2e-2 容忍度内的可读密度方程相匹配。

### 检查站1:建立正确的直接路径

先执行最简单的页面行走内核. 校对:

- 读K/V到 `block_table` 没有建造一个密密的K/V变压器,
- 忽略最后一页未使用的插槽,
- 在多个页边框和上下文长度上匹配密集的注意力,
- 支持 grouped-query attention 何时 `H_q != H_kv`.

正确性时间表可能比密集的注意力要慢. 在这
检查站,有益的结果是一个可靠的基线和一个工作运行时间
接口。

### 检查站2:设计 Decode 时间表

优化单键 decode 形状,而不是将页面翻转作为
连环 一次一次通过这些变化,一次次通过基准
一个:

1. 指定一条航道 SIMD 组合到一个头的相邻元素, 所以 K/V 负载
   可以连结,
2. 装入页表条目一次,再用于页中的所有位置,
3. 将查询和在线softmax斯状态保存在页面牌的登记册中,
4. 将部分点产品与 SIMD 减少而不是线组
   刮伤记忆和障碍,
5. 专业 `L = 1` decode 箱子,所以不会携带 prefill 控制流动,
6. 编制批次的页面元数据,每排程器步骤一次,而不是一次
   每层或注意头。

同时将写入路径分别设定基准 。 无法快速读取页面内核
恢复损失的时间到每个层之前的功能全缓存更新.

对于每个变化,解释它的目标成本:内存流量,同步,
地址计算,或发送间接费用。 仅在测量时保留更改
结果支持这一解释。

在您的解决方案中优化页面路径 Qwen 头尺寸128,
但保留一个组式排程, 而不是复制在线 softmax
单个GQA比率的重现。 保持页面间的关系
逆向,头图, 和还原可见在一个内核。

### 检查站3:评价服务系统

单靠操作员的耐用性并不能捕捉到寻呼的目的. 运行一个
输入和离开批次的终端到终端工作量,然后报告:

- 时间每 decode 每秒的步数和汇总符号,
- KV-cache存储器的峰值和接受的现场请求数量,
- 字节或用于收集和重新包装K/V的时间,
- 与溶液中密集的第二周路径相对的呼号留念间隔
  和 MLX.

保持密集的路径,作为教学的启示,这样你就可以测量在毗连时
注意得更快一点 完成的服务路线仍保持呼号:它消除了
重新包装,重新使用排程器步骤的页面,并留下更多测量的 KV
头室在固定的夹痕。 证明它承认更多的同时
请求需要一个内存封装的录入扫描。 第5天优化它
长预补时间表,而不是绕页表合同行驶。

使用配对的状态跑车,而不是预先分配的静态请求:

```bash
pdm run bench-serving-progression --offline --repeats 4 \
  --model qwen3-4b --num-seqs 16 --batch-size 4 \
  --min-input-len 128 --max-input-len 1024 \
  --min-output-len 32 --max-output-len 128 --prefill-step 128 \
  --warmup 1 --cooldown-seconds 1 \
  --json-output benchmark_results/m4-pro-qwen3-4b-week3-serving-mlx-0.32.0.json
```

将第2周密集批量重建、第3周网页存储和
密度- gather 兼容路径, 第3周直接呼叫注意 。 重装
热身和报告后的页面容量 prefill,输出,以及 decode 吞吐量
与峰值 KV 字节, 复制量, 页面再利用, 以及尾部破碎 。 那个
直接路径的四进程中位数为679.56 prefill tok/s, 41.88 产出 tok/s,
82.11 decode tok/s,和0.558请求/s. 它同步 decode 打电话接听
38.27/39.83/43.46 ms 中值/p95/最大值; 完成差距,包括
中间排程器和 prefill 工作,为39.13/224.70/240.99 ms.

```bash
pdm run test --week 3 --day 4
```

{{#include copyright.md}}
