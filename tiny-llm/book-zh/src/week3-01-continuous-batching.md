# 🚧 第 3 周第 1 天：连续批处理

>            本章正在审查中,可能有所改动。

在本章中,我们将执行 **连续批量**,保留了批次
设备上正在运行的请求,并在每一次请求中立即替换
结束。

迄今为止,每个世代循环只处理了一个请求. 可能不会提供
足够的工作来高效使用设备, 所以我们会 decode 几项请求
每一个模式调用。

静态批量可以选择5个提示并一起运行,直到每次
请求完成 。 然而,生成的序列有不同的长度. 如果四个
第五个阶段继续进行时,请求迅速完成,大部分仍保留
无法启动闲置和队列请求 。

连续批量代之以设定活动的最大数量 decode 请求。
完成后,调度器将批量槽和 KV- cache 条目指定为
等待请求。 这保持了 decode 当工作排队时, 批量地聚集。

排程器也必须交互 prefill 和 decode 工作时 我们用简单的
政策:先发制人 prefill,则 decode 一个 token 每个活动
请求。

```python
while requests_in_queue_or_in_progress:
    if prefill_request is not None:
        prefill_request.try_prefill()  # Day 1 processes the complete prompt
        if prefill_request.ready:
            if kv_cache.try_add(prefill_request):
                prefill_request = next(requests)
    if active_requests:
        tokens = decode(model, kv_cache)
        for request, token in zip(active_requests, tokens):
            request.append(token)
```

第1天的一通电话中,一个完整的提示被接受。 这样安排
政策容易检查和暴露一个重要限制:一个很长的 prefill
可以延迟每个活动请求的下一个 decode 步骤。 第2天将增加一个界限
prefill 解决公平问题的预算。

## 任务1:重复使用 RoPE 和批量请求的因果面具

```
src/tiny_llm/week2_kernels.py::FastRoPE  (reuse unchanged)
src/tiny_llm/attention.py::causal_mask   (reuse unchanged)
```

连续批量需要一次 RoPE 每个批量元素和一个因果
掩码,其查询和源长度可能有所不同。 验证这两个第二周
添加调度器之前的接口, 这样服务层可以使用一个模型
每个申请职位的合同。

校验多重抵销 RoPE 以及两种注意路径:

```bash
pdm run test --week 3 --day 1 -- -k task_1
```

## 任务2:批次 KV Cache

```
src/tiny_llm/kv_cache.py::BatchingKvCache
```

`BatchingKvCache` 每个请求缓存 decode 插槽 。 因为请求可能
有不同的序列长度,它必须结合它们的键和值
密集的对角和构造匹配 `B x 1 x L x S` 戴着面具。

```
S = max(S_i across active requests)
L = mask_length (input parameter)
request_keys: H, S_i, D
request_values: H, S_i, D
batched_keys: B, H, S, D
batched_values: B, H, S, D
mask: B, 1, L, S
```

对齐常见的每个活动请求 `S` 维度。 领先者
位置仍为零,并蒙上面具。 不活跃的插槽仍然被完全掩盖。

```python
keys_i, values_i = request_cache[i]
batched_keys[i, :, (S - S_i):S, :] = keys_i
batched_values[i, :, (S - S_i):S, :] = values_i
mask[i, :, 0:L, (S - S_i):S] = causal_mask(L, S_i)
```

您可以通过运行验证您的解决方案 :

```bash
pdm run test --week 3 --day 1 -- -k task_2
```

## 任务 3: 练习批量准备模式

```
src/tiny_llm/qwen3_week2.py  (reuse unchanged)
```

调用第2周模式,附有若干项请求,每批1项抵消,以及
面具归来 `BatchingKvCache`。练习请求加入和离开
处于不同位置。 该模式仍然是对请求的不可知性;空档所有权;
生命周期属于缓存和调度器.

你应该通过所有测试 通过运行:

```bash
pdm run test --week 3 --day 1 -- -k task_3
```

## 任务 4: 批量生成

```
src/tiny_llm/batch.py
```

首次执行 `Request.try_prefill` 通过在其中预先填充完整的提示
打电话 然后完成调度器 `batch_generate`: 移动已完成的预填
进入闲置 decode 插槽, 收集下一个 token 并冲抵每个槽,以及
删除到达EOS或 `max_seq_len`.

以 :

```bash
pdm run batch-main
```

默认情况下,此命令使用批量大小为5个且固定的 Quen3-0.6B
一组提示。 记录连续时间间隔中最长的 decode 步骤
当一个排队请求有更长的提示时。 这个间隔是基线
第2天,请使用第2天 `bench-chunked-prefill` 可公布文件的运行程序
比较:在所检查的每个时速下,其512个预算流程
64–512 切痕成块,这样一行就是可复制的第一日控制.
跑者记录准确 token 编号、产出预算、种子、工艺顺序和
解码补全漏洞,而不是依赖上面的交互提示.

{{#include copyright.md}}
