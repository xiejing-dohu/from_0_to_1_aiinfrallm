# 🚧 第 3 周第 3 天：分页 KV Cache

>            本章正在审查中,可能有所改动。

在本章中,我们将设计 **页面d KV cache**,存储抽象
注意后面的呼号。 连续批量创建许多请求拥有的缓存
不同寿命和序列长度 将每个缓存存储为一个
不断增长的变速器使每个附件都依赖于一个毗连的分布,并制作
批量施工重温历史 K/V.

固定大小的页面将序列的逻辑顺序与其物理顺序分离
放置。 运行时间可以在不移动的情况下附加、释放和再利用存储
请求的完整历史。 本章首先更改存储布局,并
使用密集的注意作为正确检查点. 第4天将阅读这些页面
直接来

**读数**

- [vLLM 注意设计](https://docs.vllm.ai/en/v0.18.0/design/paged_attention/)
- [为大语文模型服务的有效记忆管理 PagedAttention](https://arxiv.org/abs/2309.06180)

## 为什么Dense KV布局变得昂贵

现在,精神模型是这样的:

```plain
request A -> one dense KV tensor
request B -> one dense KV tensor
request C -> one dense KV tensor
```

在注意前,运行时间将它们重新包装为:

```plain
keys:   [B, H, S_max, D]
values: [B, H, S_max, D]
mask:   [B, 1, L, S_max]
```

问题是 decode 每个步骤只增加少量的新信息,但密集的布局不断重温旧的KV.

例如,如果一个请求已经拥有17个缓存符号,我们 decode 还有1个 token:

```plain
new useful work: append 1 token
dense repack view: rebuild 18 logical positions
```

对于一个请求,这是好的。 对于许多现场请求来说,运行时间花在了越来越多的时间移动先前计算过的KV,而不是做实际的模型工作.

## 页面摘要

而不是将每层的 KV 存储为一个请求的长亮度,而是将存储分为固定大小 **页次**:

```plain
key_pages:   pages with up to page_size token slots
value_pages: pages with up to page_size token slots
```

每层缓存保留一个小的页表 :

```plain
page_ids = [12, 5, 3]
context_len = 10
```

这意味着:

```plain
page 12 -> tokens 0..3
page  5 -> tokens 4..7
page  3 -> tokens 8..9
```

逻辑序列仍为长度10. 不同的是,运行时间不再被强迫作为一个毗连的角数来代表它.

模型拥有一个物理 **页 次
每个变压器层的池**。为同一层请求缓存共享其池,
而每个请求层缓存都保留自己的 `page_ids`, `page_lens`,以及
`offset`。因此,一个页码是本地到一层,匹配 K/V 存储
缓冲内核收到的注意力。

`page_size` 是物理页面容量。 未使用的尾槽不属于
逻辑顺序; `page_lens` 决定每一页的哪个前缀有效。

## 为什么固定大小页面帮助

页面抽象 给了我们两个直接的胜利:

1. 拨款a token 通常只更新池中的当前尾页.
2. 完成的请求可以返回到该层共享的自由列表中。

这是网页关注系统背后的关键内存管理想法,例如 vLLM.

## 数据结构 我们需要

### 1. `PagePool`

该模型应拥有每层一个游泳池,每个游泳池有一个自由页分配器和
平面 K/V 页面存储 :

```plain
free_pages: available page ids for this layer
keys[page_id]:   physical key page
values[page_id]: physical value page
```

请求仅在执行同一层时共享物理存储 。
第0层和第1层可能都使用页码 `[0, 1]` 因为那些IDs地址
不同的缓冲器。 将图层尺寸保持在外 `page_id` 防止
从复制或序列化的页面存储中将单键写入一层
每隔一层

支撑板应该长几何,而不是一页。
保持逻辑 `num_pages` 与体力无关 所以呼叫者仍然看到
只分配页面, 而集合增长复制旧存储对数很多
时间。

生长一个精确大小的板块 `p` 改为 `p + 1` 页数 大约
`1 + 2 + ... + p` 旧的页,在最后页数中为四进制。
从4页开始,容量增加一倍,少于
所有增长活动最终的能力。 用变压器分割板
层还防止第0层中的新页面替换存储对象
用于其他各层。 这两处变化 分期摊销 分配器复制工作 所以
大多数新页分配不复制旧页,而不更改逻辑
页面表。

执行此抽象为 `TinyKvPagedPool`.

### 2. `PagedRequestCache`

一个请求的层缓存应跟踪:

- `page_ids`
- `page_lens`
- `offset`
- `page_size`

衍生值 :

- `num_pages = len(page_ids)`
- `context_len = offset`
- `last_page_fill = page_lens[-1]` 当至少有一个页面存在时

执行请求视图为 `TinyKvPagedCache`。用来自
它不该分配自己的水池,
因为这会将一个请求从共享的页面分配符中分离出来.

创建一个 `TinyKvPagedCache` 每个变压器层。 开关不同
请求共享层池, 但是它们不共享元数据: 每个缓存拥有
其本身 `page_ids`, `page_lens`,以及 `offset`.

### 3. 尾附逻辑

当新的 K/V 到达一层时 :

1. 看那层缓存的最后一页
2. 如果有房间,请在尾页上只插入新片
3. 以其他方式分配新页并继续撰写
4. 更新缓存元数据,例如 `page_lens` 和 `offset`

这取代了序列维度上反复凸起的密度-缓存模式.

#### 使写成本与所附切片成比例

MLX 数组是功能和懒惰的评价。 写入
`pages[page_id, :, start:end, :] = values` 可能构建一个输出为
整个页面显示器; 小片输入 Python 不保证 a
切片大小的设备更新 。

执行 `paged_cache_update` 作为小扩展的原始 在你的解决方案。
它的输出别名现有的页面缓冲器,以及它的 Metal 网格
仅覆盖 `H * new_tokens * D` 要素。 页面存储是请求状态, 所以
只要缓存拥有其页面,此突变边界就明确且安全
而注意力取决于返回的数组。 仅保留全部缓存副本
当几何能力增加时。

在重建前完成每个学习者扩展集成点:

- 创建 `src/extensions/src/paged_attention.cpp` 对原始人和
  `src/extensions/src/paged_attention.metal` 为其内核,
- 登记这些 C++ 和 Metal 各自清单所列来源
  `src/extensions/CMakeLists.txt`,
- 声明 `paged_cache_update` 输入 `src/extensions/src/tiny_llm_ext.h`,以及
- 登记其 Python 绑定在 `src/extensions/bindings.cpp`.

然后重建:

```bash
pdm run build-ext
```

通过缓存界面测试此行为: 跳过尾页
边框, 增大板块, 释放和再利用页码, 比较所收集的
逻辑顺序 `TinyKvFullCache`.

## Prefill 带页

假设 `page_size = 4` 和一个 prefill 块包含6个符号 :

```plain
chunk = [t0 t1 t2 t3 t4 t5]
```

一个可能的布局是:

```plain
page 7 <- [t0 t1 t2 t3]
page 2 <- [t4 t5]        # 2 valid tokens, 2 unused slots of capacity
```

层缓存的元数据变成:

```plain
page_ids = [7, 2]
context_len = 6
```

重要财产是后期 decode token 可在页面上附加 `2` 不触摸页面 `7`.

## Decode 带页

期间 decode,每个直播请求增加一个 token 一次来

使用页面存储 :

1. 计算一键 `k` 和 `v`
2. 检查尾页是否有空格
3. 如有可能,写入该页
4. 仅在旧页满后分配新页

所以,如果 `page_size = 4` 和 `context_len = 9`:

```plain
page_ids = [12, 5, 3]
```

追加 token 9只更新最后一页,而不是重建所有早期的KV.

## 校正性检查点: 收集页面以引起注意

最干净的首次执行是 **使用稠密集合调用存储**.

这意味着:

- 每一层池的页是真理的来源,
- 层缓存停止拥有一个单质的K/V 变压器,
- 层缓存只追踪页面元数据,
- 注意力仍然得到密集的 K/V 从页面重建。

这个检查点在添加间接前隔离存储和生命周期工作
GPU 全文如下:

- 可独立测试页面分配和再利用;
- 收集的序列可直接与 `TinyKvFullCache`;
- 复制柜台确定直接翻页应删除的费用。

## 此地图将如何 `tiny-llm`

### `src/tiny_llm/paged_kv_cache.py`

添加:

- `TinyKvPagedPool`
- `TinyKvPagedCache`

保留 `TinyKvFullCache` 输入 `src/tiny_llm/kv_cache.py` 作为基准和试验
预言家

该章的执行路径为:

1. 将新的 K/V 写入层缓存的尾页或新分配的页,
2. 将图层缓存的页面收集回稠密的K/V,
3. 将这种密集的K/V输入可读的密集注意力方程。

本章在保留密集关注的同时更改存储模式
等式作为正确性预言。

### `src/tiny_llm/batch.py`

请求应当拥有每层缓存手柄,而不是长时间的密密的K/V亮度.

调度员仍应:

- 执行块 prefill,
- 提出积极的请求,
- 空闲缓存页,当一个槽完成时。

不同的是,释放一个请求现在意味着释放其层缓存所拥有的所有页面回到池中.

添加一个小 `rewind(n)` 寿命轮钩. 倒带让呼叫者删除最新的
不重建保留前缀的逻辑符号 。 它可以解开整页
并缩短最后剩余部分的有效长度
页面。 可选的投机解码章节将在
伪造的代币被拒绝。

## 设计问题

在实施之前,确保以下内容明确:

1. 这种重播应使用何种页数进行教学?
2. 我们如何代表自由页分配器?
3. 我们如何证明网页存储重构与 `TinyKvFullCache`?
4. 请求缓存手柄如何在保留自己的页面元数据的同时共享层池?
5. 我们何时实现网页写作以避免 MLX 懒惰的成长?
6. 我们如何在不复制每项分配的所有旧页的情况下发展体能?

## 任务1:设计 `PagePool`

```
src/tiny_llm/paged_kv_cache.py
```

修改 `TinyKvPagedPool.__init__`, `allocate_page`, `write_page_slice`,以及
`free_page` 在这项任务中。 对于切片尺寸的设备
写,替换第3周第3天的根 `tiny_llm_ext::paged_cache_update`,
`PagedCacheUpdate::eval_cpu`,以及 `PagedCacheUpdate::eval_gpu` 输入
`src/extensions/src/paged_attention.cpp`,并执行
`paged_cache_update_kernel` 输入 `src/extensions/src/paged_attention.metal`.
声明,具有约束力,来源/Metal 文件,而且 CMake 已经注册
已存在; 不创建第二个页缓存 API.

设计层自有的页池:

- 拥有自由页分配器,
- 储存固定大小的K/V页,
- 分配和释放页码,
- 支持将块写入页面存储,
- 支持能力几何增长,
- 仅更新生长活动之间的附属物理片段,
- 所有请求的缓存都用于该层,但不是其他层。

分别测试逻辑大小和物理容量. 分配第五页,
例如,可能创建8页的能力,但 `key_pages` 和
`value_pages` 受关注的运行时间应该只包含五个
已分配的页面编号。

## 任务2:设计 `PagedRequestCache`

```
src/tiny_llm/paged_kv_cache.py
```

修改 `TinyKvPagedCache.__init__`, `update_and_fetch`, `release`,以及
`rewind`启用 `TinyKvPagedPool.write_page_slice` 从任务 1 到
每一个物理附件。

将“一层缓存=一个密集的 KV 显示器”模型替换为:

- `page_ids`
- `context_len`
- 在固定大小页面上附加逻辑
- `release()` 在请求完成时返回页面
- `rewind(n)` 丢掉最新的 `n` 逻辑符号

## 任务 3: 添加一个 增强- Gather 兼容性路径

```
src/tiny_llm/paged_kv_cache.py
src/tiny_llm/qwen3_week3.py
```

修改 `TinyKvPagedCache.gather_dense` 输入
`src/tiny_llm/paged_kv_cache.py`外加时 `Qwen3ModelWeek3.__init__`,
`Qwen3ModelWeek3.create_kv_cache`,以及 `Qwen3MultiHeadAttention.__call__` 输入
`src/tiny_llm/qwen3_week3.py`。该检查站故意不执行
`paged_attention`; 第4天拥有该功能。

构建一个从页面中重建稠密 K/ V 的兼容路径,并将其比对 `TinyKvFullCache`.

这让我们在改变注意力路径之前 进行正确检查
以 `enable_paged_attention=False` 在此
因此,它的注意力读取了聚集的密集的角。 第4天开关
与页面表元数据及页面内核相同的模型。

通过正常生成和基准运行累计检查点
切入点 :

```bash
pdm run main --solution tiny_llm --loader week3 \
  --disable-paged-attention --model qwen3-0.6b

pdm run bench --solution tiny_llm --loader week3 \
  --disable-paged-attention --model qwen3-0.6b
```

在下一章中,我们将采取下一个步骤:我们不会在注意之前收集密集的K/V,而是通过运行时间元数据,例如: `block_table` 直接进入呼叫的注意路径。

## 什么呼号变化

苹果硅的统一内存去除了离散的-device传输边界,
但它不删除分配、分割或复制
GPU隐形堆积.
固定大小的页面仍然允许服务器重用释放容量, 不增加请求量
保留最大序列长度, 分批处理不同的请求
上下文长度。 这些是有用的生命周期机制,但固定的痕迹
衡量KV储存室,而不是接收能力。 声称更多
请求可以被接纳,需要单独的内存封盖扫描.

报告零碎化情况,并配有相应的数字和分母。 基准
在最后的直播页面中找到未使用槽数最多的快照
中,然后将总和除以所有 token 直播时段
页面在同一快照。 报告未使用的字节。 未使用
物理池容量被排除在这一部分之外,仍可见于
单独的现场网页和容量页面柜台。

```bash
pdm run test --week 3 --day 3
```

{{#include copyright.md}}
