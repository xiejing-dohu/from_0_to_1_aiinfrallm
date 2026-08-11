# 🚧 第 2 周第 2 天：基准测试与性能分析

> **现状:实验。** 见
> [第2周核查矩阵](./week2-overview.md#verification-status) (单位:千美元)
> 不断测试、在当地测量和仍在审查中的内容。

第1天给了我们一个缓存模型。 第二天,那确是一个可信赖的密室, BF16
基线:速度 prefill 和 decode 根据一个匹配的协议,什么
下一章的建筑成本应该吗? 需要制定基准。
剖面是可选的,不是先决条件或接受门。

## 缓存模式基准

优化从值得信赖的比较开始. Prefill 许多进程
立即发出指示牌; decode 通常处理一个 token 每个请求,并且是
以反复读取密度为主 BF16 投影加权数
检查站。 改变可以改善一个阶段,同时伤害另一个阶段,所以
`benches/bench.py` 报告:

- prefill 每秒符号: 提示符号除以 prefill 时间;
- decode 每秒符号: 第一次之后生成的符号 token 除以
  decode 时间

第一个生成的 token 属于 prefill。将其排除在 decode 防止
从扭曲时提示长度 decode 数字。

选择 prefill 在比较执行情况之前先处理工作量。 提示计分
需要每个位置的对数, 而服务只需要最后的提示
记录。 使用 `--prefill-logits all` 联合国
`--prefill-logits last` 为后者。 跑者对你的选择
解决方案和 MLX 一样 永远不要把最后一行从你的解决方案和
全数 MLX 运行。

"周刊2"两边的比较使用a KV cache: prefill 一次即时,
然后只传递新生成的 token 在每个 decode 步骤。 比较缓存
MLX 基线与您的解决方案重算整个前缀将测量 2
不同的算法,使下一个优化目标变得毫无意义。

### 记录匹配的基线

使用相同的模型、 快速长度、 输出长度、 设备、 热量计数
您的解决方案和 MLX:

```bash
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-4b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2 \
  --prefill-logits last

pdm run bench --solution mlx --loader week2 --model qwen3-4b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2 \
  --prefill-logits last
```

使用 `--solution tiny_llm_ref` 和相同参数比较时
您使用参考解决方案而不是 MLX.

或运行新进程的累积阶梯:

```bash
pdm run bench-week2-progression --offline --repeats 4 \
  --solution tiny_llm \
  --variant week2-kv-cache --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last --json-output week2-baseline.json
```

闲置机器基准:停止其他CPU和GPU密集型
工作量,保持电源模式和环境条件的固定,并让机器
在比较运行之前返回稳定温度。 运行每个命令数
时间,报告中位数,并包括硬件, MLX 和mlx-lm版本,
预填日志模式,并带有结果的精确模式。 依赖性升级
更改比较基线,因此重新计量 MLX 而不是怀旧
分母前进。

### 同步懒惰工作

MLX 构建懒惰的计算图表. 时间 Python 调用计量图
建筑业,不 GPU 执行。 每次重复必须评估
输出 :

```python
start = perf_counter()
output = function()
mx.eval(output)
elapsed = perf_counter() - start
```

基准还必须在温暖和定时后调用缓存释放钩
运行带有自有或共享资源的缓存执行可以返回:

```bash
pdm run test --week 2 --day 2
```

## 可选配置边界

所需的第2天工作以同步基准JSON结束. Metal
抓取,Xcode可视化, `gpudebug`,和相关特征分析
微观基准不是目前课程要求的一部分。 他们要求
macOS 27 工具发布,然后作为可选材料返回
已开放。

[可选配置通知](./week2-advanced-profiling.md) 记录此
边界。 你可以跳过它,继续直接到第3天。 没有分析工具,
跟踪、截图或微基准是先决条件或接受门。

## 为什么量化: Decode 屋顶线

那个 decode 阶段 LLM 推理通常是 **内存带宽绑定**: 每个
token 需要读取模型的重量,但工作相对较少
与他们。 使用官方的维度
[Quen3-4B配置](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json)
以计算理想的边框 :

```plain
Qwen3-4B dimensions:
  hidden size        h = 2,560
  MLP size           i = 9,728
  query width        q = 4,096
  key/value width   kv = 1,024
  layers             L = 36
  vocabulary         V = 151,936

Projection weights per layer:
  Q and O: 2 × h × q       =  20,971,520
  K and V: 2 × h × kv      =   5,242,880
  MLP:     3 × h × i       =  74,711,040
  total per layer          = 100,925,440

All transformer layers: L × 100,925,440 = 3,633,315,840
Tied vocabulary head:    V × h           =   388,956,160
Total streamed weights:                    4,022,272,000

FLOPs per token: 2 × 4,022,272,000 = 8.045 GFLOPs
```

捆绑的嵌入矩阵一次被算作词汇投影. 那个
单排嵌入式查找、正常重量、激活、KV读取和
注意工作省略。 这使得结果成为线性的上限
分层,而不是预测完整的模型吞吐量。 一个稠密的 FP16 或 BF16
重量占两个字节:

```plain
4,022,272,000 weights × 2 bytes = 8.045 GB per token
arithmetic intensity = 8.045 GFLOPs / 8.045 GB = 1.0 FLOP/byte
```

FP16 和 BF16 以不同方式划分其16个位点: FP16 给出更多位数到
说到这里 BF16 给发音者更多的位数。 影响数字
范围与精度,但不是这种带宽计算。 课程使用 BF16
用于激活和输出。

| 密集重量格式 | 每个重量的位数 | 每重量字节 | 流式重量字节/ 每 token | 重量计算强度 |
|---|---:|---:|---:|---:|
| FP16 | 16 | 2 | 8.045 GB 数字 | 1.0 FLOP/字节 |
| BF16 | 16 | 2 | 8.045 GB 数字 | 1.0 FLOP/字节 |

这是改进的基线:两种密集格式必须流出大约8GB
投影加权数以生成一个 token保存匹配的基准结果,
然后继续学习[第 3 天](./week2-03-quantize-model.md)，届时模型会保留
装入重量, 替换实时投影路径, 重运行相同
基准。

{{#include copyright.md}}
