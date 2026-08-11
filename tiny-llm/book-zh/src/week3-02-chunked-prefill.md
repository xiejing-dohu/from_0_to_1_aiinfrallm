# 🚧 第 3 周第 2 天：分块 Prefill

>            本章正在审查中,可能有所改动。

长时间的提示可以在激活时垄断设备 decode 请求等待
他们的下一个 token切换 prefill 给每个排程器迭代一个快速键
预算,限制多长时间 decode 工作可以推迟。

调度政策如下:

```plain
admit at most prefill_max_step prompt tokens
decode one token for every active request
repeat until the queue and active batch are empty
```

## 任务1:边界 Prefill 工作

更新 `Request.try_prefill` 输入 `src/tiny_llm/batch.py` 选择一个提示
切片,用切片的绝对偏移调用模型,并标记请求
只有在处理完全部提示后才能准备。

```python
for start in range(0, len(prompt_tokens), prefill_max_step):
    chunk = prompt_tokens[start : start + prefill_max_step]
    model(chunk, offset=start, cache=cache)
```

最后一个块可能比配置的预算小。 测试提示缩短
一个块,一个块,一个块 token 长于块。

## 任务 2: 构建矩形 Causal 面罩

当缓存已保存时 `S - L` 符号和块贡献 `L` 新设
标志,面具是 `L x S`。每个查询都可以处理旧的前缀和
更早的位置在自己的块。

对于一个五孔的前缀和一个三孔块,面具是 `3 x 8`:

```plain
0  0  0  0  0  0  -inf  -inf
0  0  0  0  0  0     0  -inf
0  0  0  0  0  0     0     0
```

使用绝对缓存偏移 RoPE 和 `S - L` 作为因果对角
页:1 比较块 prefill 用一发对数 prefill 记录

## 任务 3: 在块之间实现物质化

MLX 懒惰。 多次扩展未评价缓存会生成一个长图
可以增加内存使用量。 调用每层缓存 `materialize()` 后钩
块,所以下一个排程器的迭代从实现状态开始 。 一个稠密的
缓存评价它的密钥/ 值 Tuple; 页面缓存评价页面池
存储时不先将其聚集到密集的抗震器中。

钩子是缓存寿命周期的一部分 而不是调度器的存储
逻辑 这样可以让调度员不用检查即可使用密密和调用缓存
他们的内部代表。

## 任务4:衡量公平权衡

运行同一请求的追踪数 `prefill_max_step` 数值。 报告共计
吞吐量和连续时间间隔最长 decode 步骤。 小一点
块通常提高公平性,但添加排程器和发射管理费用;选择一个
而不是将一个块大小当作
通用。

```bash
pdm run test --week 3 --day 2
pdm run batch-main

pdm run bench-chunked-prefill --offline --model qwen3-0.6b \
  --prefill-steps 32 128 512 --num-seqs 8 --batch-size 4 \
  --min-input-len 64 --max-input-len 512 \
  --min-output-len 32 --max-output-len 32 \
  --warmup 1 --repeats 4 --cooldown-seconds 1 \
  --json-output benchmark_results/m4-pro-qwen3-0.6b-week3-chunked-prefill-mlx-0.32.0.json
```

选中的微量使用种子 0 和相同的 32 个token 输出预算
请求。 每个块大小按前序运行两次,反序运行两次
在新的过程中。 JSON每时每刻 token 编号,按要求
产出预算,以及他们的Canonic SHA-256检查和。

解码补全间隙是连续两个时间间隔之间的间隔
同步 decode 通话时至少一个 decode 请求仍然有效。 这个
因此包括干预 prefill 和排程器工作;空闲时间没有
decode 请求不包括在内。 在测量的M4 Pro上,四进程中位数
原为:

| Prefill 预算 | 产出 tok/s | 请求/请求 | Decode 步骤 p95 | Decode p95/最大间距 |
|---:|---:|---:|---:|---:|
| 32 | 105.47 | 3.296 | 17.52 ms | 30.39 / 32.47 ms |
| 128 | 144.91 | 4.528 | 18.78 ms | 46.52 / 48.80 ms |
| 512 | 157.00 | 4.906 | 19.57 ms | 76.04 / 122.16 ms |

512号token列车是这一追踪的全速第一日控制器. 减少
预算使p95的完成差距单调缩小,而32位
预算放弃了大量的吞吐量。 该课程使用128作为计量
以妥协方式解决这一工作量,而不是作为普遍的最佳办法。

{{#include copyright.md}}
