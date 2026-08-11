# 🚧 第 2 周第 3 天：模型量化

> **现状:实验。** 见
> [第2周核查矩阵](./week2-overview.md#verification-status) (单位:千美元)
> 不断测试、在当地测量和仍在审查中的内容。

第2天建立了同步密度 BF16 基线。 第3天减少预测
装满W4A16重量的重量流量,执行 Metal 运算符
直接消耗,把这些操作员接进直播模型,然后重新运行
同样的基准。 包装存储或孤立的快速内核未完成 :
缓存模型必须使用量化路径。

**读数**

- [模型压缩和量化](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [MLX 扩展发展指南](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
- [量化 Matmul 打开 GPU (视频)](https://www.youtube.com/watch?v=jYCxVirq4d0)

## 调试 Metal 没有 CPU 双

A C++ CPU 版本是可能的,但不需要。 使用此三级验证
梯度替换为 :

1. 将方程式写入 Python 与 `mlx.core`这是语义论
2. 将其译为故意简单 Metal 内核,通常有一个
   用于输出元素的线程 。
3. 优化验证的 Metal 内核为 SIMD 组合、矢量载荷,或
   SIMD-组矩阵操作.

将每一关与上面的关卡进行比较. 不要调试优化的
只比较完整模型文本输出的内核.

### 使失败小而同步

从预期值易于检查的确定性固定装置开始:
零, 1, 斜坡, 类似身份的重量, 和一个固定的随机种子。 演习a
小对齐形状,然后是尾部形状。 例如,测试8和10行
一个8-排的瓦片,或一个32-排的块的序列长度为32和35。

MLX 执行是懒惰的,所以在操作符下直接强制评估
测试。 转换延迟编译或 GPU 执行失败为失败
负责呼叫网站:

```python
expected = python_reference(*inputs)
actual = metal_operator(*inputs)
mx.eval(expected, actual)

assert actual.shape == expected.shape
assert actual.dtype == mx.bfloat16
assert mx.allclose(actual, expected, rtol=2e-2, atol=2e-2).item()
```

在检查算术前先检查包装边界. 保持呼声
级别,形状, dtype和连续假设 Python 或 C++,并核实
编码缓冲指数与 Metal 函数签名。 然后分类
失败 :

- 管道创建错误通常指内核名称、专门化,或
  Metal 汇编错误;
- 执行或地址错误通常是指网格、边界检查、脚步或
  缓冲绑定错误;
- 有限但不准确的结果通常意味着索引、减少、面具、
  脱量化,或累积器更新是错误的.

对于数字不匹配,暂时简化调度. 指派一个输出
至一个线程,去除合作负载,并比较中间体,如
减量重量组,部分点产品,或在线 softmax排. 页:1
小型只调试输出缓冲器往往比从每个程序打印更有用
GPU 线程。 一次恢复一次优化并重运行对齐和
每次改变后进行尾形测试.

## 代表重量, 位数较少

**量化** 代表带有小值的浮点权重
整数代码本加上大约重建
原始价值。 此课程使用 **仅重 4 位量化**:

- **W4** 表示每个逻辑权重由一个4位码代表.
- **A16** 表示激活和输出仍为16位浮点.
- 由此产生的路径叫做 **W4A16**本课程使用 BF16 联合国
  激活、尺度、偏见和产出。

只有16个可能的代码,重建后的权重大约为原始
数值。 代表人数较少,则在数字上比较精确。
记忆交通.

内核不会变成密集的 BF16 重量矩阵。 每个都拆开
4位码, 重建寄存器中的重量, 然后立即乘数
由相应者决定 BF16 启动

### 组- Wise Affine 量化

我们不是对整个重量矩阵应用一个比例表,而是将每行划分
输入 **组** 并独立地量化每个群体。 当地比额表和偏见
保存关于每个组的重量分布的更多信息。

重量矩阵 $W$ 形状 $(K, N)$,将每行分为大小组
$G$. Quen3-4B号 MLX 本课程使用的4位检查站有一个固定组
大小为128:

```plain
Logical weight matrix W: K × N

Group size: G = 128
Number of groups per row = N / G

For each stored group of G consecutive values in a row:
  1. Unpack each unsigned 4-bit code q in [0, 15]
  2. Load the group's stored scale s and bias b
  3. Reconstruct each value as q * s + b
```

### 重建一个存储组

检查站已经装有包装的代码及其附属参数.
对于一个无包装的未签名代码 $q$缩放 $s$ 和偏见 $b$
直接 :

$$
\hat{w} = q s + b
$$

代码没有签名,但存储的尺度已经签名. 正比例图
代码0到下端,代码15到上端。 负数
缩放 逆向 : 代码 0 是上端点, 代码 15 动作
向下端移动 两个方向都发生在发运的Quen3-4B中 MLX
检查站,不要重算 `scale` 和 `bias` 从假设分钟/最大值
定向。

例如,这两个存储的参数对重建同一个端点
以相反代码顺序排列的区域 :

```plain
positive orientation: scale =  0.0867, bias = -0.5  => q=0 is -0.5, q=15 is about 0.8
negative orientation: scale = -0.0867, bias =  0.8  => q=0 is  0.8, q=15 is about -0.5
```

所有要求的量子-量子测试使用情况 `group_size = 128` 和 BF16 比额表,
偏见、激活和产出。 使这些抗震器规范化 BF16 在你身边
溶液的模型加载器,所以以后每个内核都接收一个模型 dtype.

### 包装存储布局

4位代码被包装为紧凑存储和高效访问:

```plain
Logical weight matrix: K × N
Dense BF16 storage: K × N bfloat16 (2 bytes each) = 2KN bytes
W4 code storage: K × N int4 (0.5 bytes each) = 0.5KN bytes

Packing: 8 × 4-bit values fit in one uint32 (32 bits)

Packed codes shape: K × (N / 8) uint32
Scales shape: K × (N / G) bfloat16
Biases shape: K × (N / G) bfloat16
```

连续8个4位值的示例包装 `[a, b, c, d, e, f, g, h]`:

```plain
uint32_value = (h << 28) | (g << 24) | (f << 20) | (e << 16) |
               (d << 12) | (c << 8)  | (b << 4)  | a

Unpacking:
  a = (uint32_value >> 0)  & 0xF
  b = (uint32_value >> 4)  & 0xF
  c = (uint32_value >> 8)  & 0xF
  ...
  h = (uint32_value >> 28) & 0xF
```

## 重审 Decode 屋顶线

包装的代码不是W4的全部代表. 每组128人
重量也存储一个 BF16 缩放和一个 BF16 偏差 :

```plain
bytes per W4 weight = 0.5 + (2 + 2) / 128 = 0.53125 bytes
streamed W4 bytes   = 4,022,272,000 × 0.53125 = 2.137 GB per token
arithmetic intensity = 8.045 GFLOPs / 2.137 GB = 3.765 FLOPs/byte
```

现在W4可以添加到密度比较中:

| 重量格式 | 值位 | 每128重量元数据 | 有效字节/ 重量 | 流式重量字节/ 每 token | 重量计算强度 |
|---|---:|---|---:|---:|---:|
| FP16 | 16 | 无 | 2 | 8.045 GB 数字 | 1.0 FLOP/字节 |
| BF16 | 16 | 无 | 2 | 8.045 GB 数字 | 1.0 FLOP/字节 |
| W4 | 4 | 一个 BF16 缩放和一个 BF16 偏差 | 0.53125 | 2.137 GB 数字 | 3.765 FLOPs字节 |

代表性较小使预测重量流量减少了3.765×.
这个比例是一键的带宽上限 decode而不是承诺
相同的端对端加速。

### 理论 Decode 跨越苹果硅的屋顶线

苹果公司发布统一的元件带宽,但不具有直接可比性 BF16
GPU TFLOPS图. 因此,可以不计算带宽屋顶线
假设计算上限:

```plain
ideal tokens/s = advertised memory bandwidth / streamed weight bytes per token
```

该表采用每个命名芯片的最高带宽配置. GB为
十进制,符合苹果公司的规格. 这些是理论上的上限,不是
基准结果。

| 芯片 | 带宽 | FP16/BF16 屋顶线 | W4 屋顶线 |
|---|---:|---:|---:|
| M1 专业 | 200 GB/s (每秒) | 24.9 tok/s | 93.6 tok/s |
| M1 最大键 | 400 GB/s (每秒) | 49.7 tok/s | 187.2 tok/s |
| M1 超强 | 800 GB/s (每秒) | 99.4 tok/s | 374.4 tok/s |
| M2 专业 | 200 GB/s (每秒) | 24.9 tok/s | 93.6 tok/s |
| M2 马克斯 | 400 GB/s (每秒) | 49.7 tok/s | 187.2 tok/s |
| M2超强 | 800 GB/s (每秒) | 99.4 tok/s | 374.4 tok/s |
| M3 专业 | 150 GB/s (单位:千兆克) | 18.6 tok/s | 70.2 tok/s |
| M3 马克斯 | 400 GB/s (每秒) | 49.7 tok/s | 187.2 tok/s |
| M3 超强 | 819 GB/s | 101.8 tok/s | 383.3 tok/s |
| M4 专业 | 273 GB/s (单位:千兆克) | 33.9 tok/s | 127.8 tok/s |
| M4 马克斯 | 546 GB/s | 67.9 tok/s | 255.5 tok/s |

广告带宽来自苹果公司的规格
[M1 Pro和最大](https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/),
[M1超强](https://www.apple.com/newsroom/2022/03/apple-unveils-m1-ultra-the-worlds-most-powerful-chip-for-a-personal-computer/),
[M2 Pro和最大](https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/),
[M2超强](https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/),
[M3 Pro和最大](https://support.apple.com/en-us/117736),
[M3 超强](https://www.apple.com/mac-studio/),以及
[M4 Pro和最大](https://support.apple.com/en-us/121553)。苹果当前的Mac
工作室将M4 Max与M3 Ultra配对,因此没有M4 Ultra行.

这些值假设峰值广告带宽,每个投影一个读
重量,没有其他交通或工作。 实际吞吐量较低,因为
完整的模型还读取激活和KV,发射其他操作器,以及
不持续维持峰值带宽。 那个
[业绩附录](./appendix-performance.md) 记录计量结果
与这一理论练习分开。

此顶线描述单键 decode时, `M = 1` 每条流线
重量为一行激活 Prefill 跨越多个区域重复使用每个重瓦
行,增加算术强度. 因此,它需要一个矩阵表;
联合国 decode 带宽比不应作为 prefill 预言

## 量化矩阵乘法

### 数学公式

标准矩阵乘法 $C = AB^T$ 其中:

- $A$: 形状 $(M, N)$bfloat16(活动)
- $B$: 形状 $(K, N)$, **数量** 改为单位4(重量)
- $C$: 形状 $(M, K)$同为 16 位 dtype 作为 $A$ (产出)

每个元素 $C[i, k]$ 计算为:

$$
C[i, k] = \sum_{j=0}^{N-1} A[i, j] \times B[k, j]
$$

随着数量化, $B[k, j]$ 现作为:

$$
B[k, j] = B_{\text{quantized}}[k, j] \times \text{scale}[k, g] + \text{bias}[k, g]
$$

地点 $g = \lfloor j / G \rfloor$ 是组索引。

替换:

$$
C[i, k] = \sum_{g=0}^{N/G-1} \sum_{j'=0}^{G-1} A[i, g \times G + j'] \times (B_{\text{quantized}}[k, g \times G + j'] \times \text{scale}[k, g] + \text{bias}[k, g])
$$

重新排列 :

$$
C[i, k] = \sum_{g=0}^{N/G-1} \left( \text{scale}[k, g] \sum_{j'=0}^{G-1} A[i, g \times G + j'] \times B_{\text{quantized}}[k, g \times G + j'] + \text{bias}[k, g] \sum_{j'=0}^{G-1} A[i, g \times G + j'] \right)
$$

尺度和偏差在一个组内是恒定的,所以计算可以重复使用
它们跨越了这个群体中的所有价值观。

### 计算流程

```plain
Input:
  A: M × N (bfloat16 activations)
  B_quantized: K × (N/8) (uint32, packed weights)
  scales: K × (N/G) (bfloat16)
  biases: K × (N/G) (bfloat16)

Output:
  C: M × K (bfloat16)

For each output element C[i, k]:
  sum = 0  # float accumulator
  for each group g in 0..(N/G - 1):
    scale = scales[k, g]
    bias = biases[k, g]

    # Process G values in the group (G/8 uint32 packs)
    for each pack p in 0..(G/8 - 1):
      packed_value = B_quantized[k, g*(G/8) + p]

      # Unpack 8 × 4-bit values
      for bit_offset in [0, 4, 8, 12, 16, 20, 24, 28]:
        quantized = (packed_value >> bit_offset) & 0xF
        b_value = quantized * scale + bias
        a_value = A[i, g*G + p*8 + bit_offset/4]
        sum = sum + a_value * b_value

  C[i, k] = bfloat16(sum)
```

## 任务1:执行量化线条和嵌入

```
src/tiny_llm/quantize.py
src/tiny_llm/embedding.py
```

修改这些精确启动函数 :

- `QuantizedWeights.from_mlx_layer`, `dequantize_weights`,以及
  `quantized_linear` 输入 `src/tiny_llm/quantize.py`;
- `QuantizedEmbedding.__call__` 和 `QuantizedEmbedding.as_linear` 输入
  `src/tiny_llm/embedding.py`.

启动码提供 `QuantizedWeights`,用于量化的容器
矩阵及其分解参数:

| 外地 | 形状 | 说明 |
|-------|-------|-------------|
| `weight` | $(K, N/8)$ uint32 | 包装量化重量. 每个 uint32 存储8个连续的4位值。 |
| `scales` | $(K, N/G)$ 浮点16 | 存储的每组标定值因子 。 符号决定了哪个终点映射到低码. |
| `biases` | $(K, N/G)$ 浮点16 | a 包括未缴摊款,不论能否收到。 代码 0 重建到此值 。 |
| `group_size` | 单位 | 相同比例/比值的连续值数。 对于 Qwen3 MLX 这里使用的4位重,这是 `128`. |
| `bits` | 单位 | 量化位宽度( 典型的 4, 表示值在幅度内) $[0, 15]$) |

内容 `from_mlx_layer` 方法提取这些字段 MLX 整数层
装入模型时。

下一步,执行 `quantized_linear`,一个包裹周围 `quantized_matmul` 与
与标准相同的输入惯例 `linear` 函数。 你会执行
`quantized_matmul` 在下一个任务中。

留着 token 嵌入的表也是量化的。 添加一个 `QuantizedEmbedding`
包装有两种调用模式:

- `embedding(input_ids)` 进行排行检查。 收集匹配的包装
  权重、天平和偏见 各解装 `uint32` 有轮班和面具,
  重复每个组在128个值上的大小和偏差,并计算
  `q * scale + bias` 有基础 `mlx.core` 数组操作。 不要打电话
  `mx.dequantize`。将这个解包逻辑放入 `dequantize_weights(...)` 这样
  嵌入及其直接测试有一个明确的执行。
- `embedding.as_linear(h)` 是绑定的输出投影。 用
  `quantized_linear(h, embedding_weight)` 因此它用你的量化 matmul 路径
  而不是完全实现 `vocab_size x hidden_size` 表单。 此路径
  一旦量化, 就开始工作 matmul 内核在下一个执行
  任务。

## 任务2:定义量化 Matmul 初级

```
src/extensions/src/tiny_llm_ext.h
src/extensions/bindings.cpp
src/extensions/src/quantized_matmul.cpp
src/extensions/CMakeLists.txt
```

启动器已经包含声明, 失败关闭源根, 绑定,
和建立注册。 留着 C++ 宣言和定义
`tiny_llm_ext` 命名空间和修改这些精确函数:

- **`tiny_llm_ext.h`** - 阅读第2周第3天 `quantized_matmul(...)`
  声明和声明 `QuantizedMatmul` 原始接口; 将其签名保存在
  与绑定同步。
- **`bindings.cpp`** - 核查现有的 `m.def("quantized_matmul", ...)`
  条目; 不创建第二个约束。
- **`quantized_matmul.cpp`** - 替换正文
  `tiny_llm_ext::quantized_matmul(...)` 用于验证
  输入, 确定输出形状, 返回懒惰 `mx::array`拒绝 CPU
  明确载于 `QuantizedMatmul::eval_cpu(...)`.
- **`CMakeLists.txt`** - 核查现有的 `quantized_matmul.cpp` 来源
  注册; 不添加重复。

延期 API 是基础设施:它允许 `mx.array` 图表节点时间表
联合国 Metal 在下一个任务中循环。 MLX 拥有数组寿命,并且
命令编码器,但不提供定量乘法。

构建扩展以获取声明、约束和注册不匹配。
以下重点测试检查任务1 Python 包装; 原始变成
执行后可运行 Metal 任务3:

```bash
pdm run build-ext
pdm run test --week 2 --day 3 -- -k task_1
```

## 任务3:执行 Metal 矩阵产品

在写第一篇之前 Metal 内核,了解执行模式. Metal
组织 GPU 在四个嵌入式瞄准镜中工作:

- **莱恩 (线条).** 最小的单位。 每个车道执行相同
  带有自己注册文件的指令流。 内巷a SIMD 组
  可以通过 `simd_` 操作。
- **SIMD 组(warp/子组)。** 一套固定大小的车道( 苹果32)
  GPUs),在锁步执行. `simd_sum`, `simd_shuffle`,以及
  `simdgroup_matrix` 在这一范围内开展业务活动。 页:1 SIMD 组不能
  直接与另一个登记册共享 SIMD 组合到同一线组。
- **线索组(块).** 收集资料 SIMD 预定的小组
  一个 GPU 核心键。 线索组共享线组内存( 明确分配)
  与 `threadgroup` 地址空间和同步
  `threadgroup_barrier`) ). 网格是一个1D/2D/3D系列的线组.
- **线网.** 全部工作都派来了 `dispatchThreadgroups` 发射网格
  线组; GPU 将它们排入现有核心。 增加
  网格的线组计数可以显示更独立的工作,但更细
  分区也可以重复读取或要求部分结果合并。

将两个发射把手分开. 更多 SIMD 在一个线程组内的组合添加
线程和可以提高寄存需求;它们增加了线组-记忆使用
仅当调度分配每个组或瓦片的共享存储时。 无论是
资源可以减少驻地线程组的数量。 更多线程组
。 不变
保证较高的吞吐量。

使用所需的双 SIMD 组备忘表作为 Qwen 起点,那么
每个线组的基准二、四、八和十六组如下。
单独更改网格分区, 所以每个启动的测量答案
把手伸出来

```
src/extensions/src/quantized_matmul.metal
src/extensions/src/quantized_matmul.cpp
```

修改这些精确启动函数 :

- `QuantizedMatmul::eval_gpu` 输入 `quantized_matmul.cpp`;
- `quantized_matmul_vanilla_w4a16_g128` 和
  `quantized_matvec_x4_fast_w4a16_g128` 输入 `quantized_matmul.metal`;
- `quantized_matmul_vanilla` 和 `quantized_matvec_custom` 输入
  `src/tiny_llm/quantize.py` 用于明确的比较路径。

写 Metal 内核和连接 `eval_gpu` 给他们。 那个 Python
`quantized_matmul` 包装总是发送您执行的原始
GPU; 解决方案中所需的路径从不经过
`mx.quantized_matmul`.

分两个阶段进行。 他们暴露了相同的数学,但时间表
不同的形状不同:

1. **香草 matmul:** 一个 Metal 线程计算一个输出元素。 这是
   直线 GPU 翻译上面的计算流程和一个可检查的
   提控器
2. **SIMD 备忘 :** (单位:千美元) decode, SIMD 通道合作减少一个
   激活行并一起计算多个输出列.

在这里, `M` 平整每个线索后的激活行数
维度。 第3天使用此明确发送:

| 激活行 | 内核 | 在这个检查站的作用 |
|---:|---|---|
| `M <= 8` | SIMD 备忘 | 优化路径 decode 以及其他非常小的矩阵投入。 |
| `M > 8` | 香草 matmul | 第一正确性问题 prefill 路径; 第6天将其替换为合作的平板内核。 |

截断不意味着 SIMD 内核扩展以覆盖更大的 `M`两者
路径是单独的调度: 第3天优化矢量形状 decode
瓶颈和叶子矩阵形状 prefill 后一个基准可见
选择。

保持香草功能可调用为 `quantized_matmul_vanilla`一个
当可以直接比较到
它取代了它的执行。

### 第一阶段:香草 Matmul

从输出行上的二维网格开始 `i` 和产出栏 `k`.
每个线程都行走 `N` 输入值,从每个输入值中拆卸8个整数
`uint32`,应用组尺度和偏差,并累积一个 `C[i, k]` 输入
浮点32. 此内核重复激活负载, 不共享工作, 而是它的
控制流镜像方程,使其成为有用的调试控制。 那个
Python `mlx.core` 等式仍然是两者的正确性 Metal
时间表。

保留矩阵形状的香草内核 prefill 本章; 第6天
重新审视这种工作量与合作的纠缠。

### 第二阶段: SIMD 马特韦克

Decode 一般情况下 `M = 1`; 8x8 矩阵牌将留下大多数行的空格。
相反,一个 SIMD 组减少输入尺寸和使用 `simd_sum` 改为
将车道与地方部分金额合并。 开始于每组两个输出列
可检查的时间表。 对Qwen3-4B检查站,然后评估一个四栏
路径,每个车道装载两个相邻的包装词,或16个激活词,以及
在四种产出中重复使用。

优化后的路径还使用affine 身份

$$
\sum_j a_j(sq_j+b) = s\sum_j a_jq_j + b\sum_j a_j
$$

避免将偏差分别应用到每个未打开的值。 规模
一次通过16位面罩读出4个装箱的整数
避免每个重量和输出行的转变。 这增加了活体蓄积器,
而不是假设整数更少
指示必须更快。

### 告诉 SIMD 时间表

将输出宽度、线程组大小和共享记忆再利用作为基准
变量。 使用此以文为重点的起点 :

- 平整所有主要激活维度 `M`,
- 使用自定义的备忘式,当 `M <= 8` 还有香草 matmul 何时 `M > 8`,
- 计算每列四个产出 SIMD 组合并装入两个相邻的打包单词
  每个车道,
- 二号发射 SIMD 组合,或八个输出行,每个线组。

这些阈值是测量的起点,而不是数学要求.
在调度器中保持可见,然后一次改变一个选择. 比较
2、4和8个产出栏 SIMD 组合。 增加栏数
激活再利用,但也延长累积器寿命,提高记录仪
压力。 比较二、四、八、十六 SIMD 每个线组的组合。
更多群体暴露出额外产出,但可能重复激活读取和
减少居住。

作为完整时间表的一部分,评估轴线重排。 下层
指令计数只有在寿命更长的激活和输出时才有用
累积器不会减少占用。 同步选择时间表
整个模型 decode 基准,而不是指令计数估计数。

界定一个连成一行的Python 至扩展合同,涉及比例尺、偏差,
激活器和装满的重量 调用 `mx.contiguous` 一次在那条边界上
校验布局 C++ 在编码内核之前是原始的. Metal
接收原始缓冲而不是隐含的数组步骤,所以布局是
正确性和性能条件。

使用直接激活读取您的内核 。 单排激活是
小型且缓存方便, 同时在线组内存中设置它会增加一个屏障
每一个投影。 如果测试共享中程为阴极,请报告
整个模型的结果,并且只有在再利用超过同步时才能保留.

### 核心要求

执行两个所需的内核布局 `quantized_matmul.metal`:

- 首先，实现朴素的“每个输出对应一个线程”的矩阵网格。
- 对于 `M <= 8`,指定一个 SIMD 组合到输出平板。 合作减少
  输入尺寸,并计算每组数个输出列。
- 对于 `M > 8`,发送香草矩阵网格。 不随行循环
  SIMD Matvec 时间表; 第6天介绍平板 prefill 时间表。
- 所需的内核支持 `bfloat16_t` 投入和产出。 第2周
  检查站不添加第二个模型存储 dtype.
- 应用本章前面定义的分组解码循环:
  - 超过128个值组。
  - 从每个包中解开整数 `uint32`.
  - 将每个值与 `q * scale + bias`.
  - 累积产品 `float`,然后将结果投向内核 dtype.
- 增加边界检查(`i < M`, `k < K`在写入输出前 。

自定义内核只需要支持 `bits = 4` 和 `group_size = 128`启用
要计算组大小 `groups_per_row` 和负重的折价。
说明所需条件 Metal 内核 `bfloat16_t` 选择其中
`eval_gpu`。如果保留可选选项 `half` 专业,不要让它进入
模型发送在您的解决方案。

### GPU 调度

完成 `eval_gpu` 输入 `quantized_matmul.cpp` 如下 `axpby`'s GPU
发送模式 :

1. 获取 Metal 设备与命令编码器。
2. 装入量化 matmul 匹配输出的内核 dtype 从 Metal
   图书馆。
3. 固定输入和输出缓冲器和尺寸常数(`M`, `N`,
   `K`) ). 缓冲命令必须匹配内核签名.
4. 选择矩阵向导布局 `M <= 8`; 否则选择香草
   矩阵布局. 保持两条路径的清晰度,以便直接比较。
   计算 SIMD 线程组配置和平板输出列
   因此,包装的输入值和激活可以被重用. 用四栏,
   两包内核和两包内核 SIMD 组合。
5. 与 `dispatchThreadgroups`.

您可以通过运行测试您的解决方案 :

```bash
pdm run build-ext
pdm run test --week 2 --day 3 -- -k gpu
```

直接测试包括: `M = 1` 和 `M = 8`香草 香草 matmul 时间
`M = 128`,并将它们与 MLX 预言家 甲骨文检查结果;
它不是正在测试的执行。

## 任务4:在继续之前进行整合

```
src/tiny_llm/qwen3_week2.py
```

修改 `Qwen3ModelWeek2.__init__`, `Qwen3MultiHeadAttention.__call__`,
`Qwen3MLP.__call__`,以及 `Qwen3ModelWeek2.__call__` 在这项任务中。 这些是
准确的点数 装载量化重量, 替换密度预测,并保持
只有请求的对数行。

将量化矩阵乘法纳入第2周 Qwen3 模型这样
线性地层在推理中一直被量化。

更改重量类型从 `mx.array` 改为 `QuantizedWeights` 百分比
注意力预测`wq`, `wk`, `wv`,以及 `wo`)和MLP预测(`w_gate`,
`w_up`,以及 `w_down`) ). 替换 `linear(x, w)` 与 `quantized_linear(x, w)`输入
第2周的模型加载器,使用 `QuantizedWeights.from_mlx_layer(...)` 改为
实现一个16位矩阵。 保持第1周模型的边界不变;
层仍然期望平整 `mx.array` 重量。

对于嵌入,电线 `QuantizedEmbedding` 从任务 1 到装入器: 装入
`embed_tokens` 与 `QuantizedWeights.from_mlx_layer(...)` 并传递给
`QuantizedEmbedding`如果模型有单独的 `lm_head`,保持头像
`QuantizedWeights` 并应用它 `quantized_linear`; `lm_head` 是一个
投影,而不是嵌入式查看。

将每个加载层的天平和偏差规范到 BF16. 需要天平,
偏差和激活以匹配和返回 BF16。如果输出为 `nan` 或
否则无效, 请检查a dtype 先错位.

也保留四分层的参数 模型应该通过
`w.group_size` 和 `w.bits` 以验证课程
假设: `group_size = 128` 和 `bits = 4`.

您可以通过运行测试您的解决方案 :

```bash
pdm run test --week 2 --day 3

pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint quantized-matvec --model qwen3-4b
```

您也可以设定您的解决方案 :

```bash
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint quantized-matvec --model qwen3-4b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2
```

运行相同的命令 `--solution tiny_llm_ref` 比较一下
参考解决方案。

香草基质产品仍可称为可检验产品 Metal 控制系统,
不过 Python `mlx.core` 等式是正确性预言,只有 SIMD
matvec 集成为 decode.

## 验证完整模型中的量化

在继续前,确认被量化的matvec内核实际上叫做
在模型推理时, 不只是在孤立地登记和测试。

> *** 接受标准。** 你的检查站在模特儿出门前是不完整的
> 投影调度器被连接到您的自定义原始 。 解码形状的工作
> 必须经过 `quantized_linear` → `quantized_matvec_custom` – 报告
> 扩展原始 → Metal 马特维克。 矩阵形状的工作必须经过
> `quantized_linear` → `quantized_matmul` 它的延伸原始 Metal
> 矩阵表。 使用源追踪通过您完成的分支
>调度器和型号布线. 提供的测试验证了包装的模型状态
> 和直接运算符,而匹配的基准报告则完成模型
> 吞吐量; 既不证明现场 Metal 管道身份本身。 使用一个
> 发送分支的直接源跟踪,然后处理吞吐量
> 作为单独结果进行比较。

测量累积模型和实际投影形状:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-kv-cache --variant week2-quantized-matvec --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last

pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
  --section decode-projections --context 128
```

附上排行前/排行后的完整模型,每个投影间隔表,
和直接的追踪。 首先需要明确 decode 增加额
`kv-cache`。然后比较每个投影 MLX 在相同的形状。
预测可能仍然是最大的绝对类别,因为模型的运行
一旦他们的操作者在接近时, MLX那个酒吧是
不再是最大的可移动缺口。

继续到第4天 直到正确性测试通过 源头追踪证明
运行模型选择预定的矩阵和matvec分支,匹配的
模型运行改进 decode 结束 `kv-cache`,预测表接近
MLX 在相同的形状。 如果预测比较还远远落后,请保留
改编曲子。 一旦空隙缩小,第4天变成反复发生
围绕这些预测实现正常化、定位和启动工作。

> **选择性貌相证据.** 内核组重播或操作员属性
> 能够证实这一过渡,但两者都不进展。 那个
> [参考检查站](./appendix-performance.md#day-3-keep-weights-packed)
> 包括上述模型和预测测量。

{{#include copyright.md}}
