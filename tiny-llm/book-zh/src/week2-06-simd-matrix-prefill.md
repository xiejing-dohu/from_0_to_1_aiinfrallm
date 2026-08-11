# 🚧 第 2 周第 6 天：SIMD-Matrix Prefill

> **现状:实验。** 见
> [第2周核查矩阵](./week2-overview.md#verification-status) (单位:千美元)
> 不断测试、在当地测量和仍在审查中的内容。

第5天结束, 将基准从单键切换 decode 改为多键
prefill其来源的痕迹显示,128token prefill 仍然使用第3天
校正性- 第一个 Vanilla 定量矩阵路径 `M > 8`第6天替换
,然后测量完整的模型和真实的
用于决定新路径是否保留。

MLX 仍然是外部性能分母; 您中的 SIMD 矩阵路径
解决方案继续称为 C++/Metal 原始的您执行每个
预测。

> **选择性貌相证据.** 经核对的依赖感归属和
> 该
> [参考-解决归属](./appendix-performance.md#checked-operator-attribution-that-selects-each-chapter)
>解释为什么预测是参考解决方案的下一个目标. 他们不是
> 需要学习者输出, 不打开本章 。

执行工作仍然刻意缩小:

- W4A16重量,4位和组尺寸128;
- BF16 a. 激活、量化参数和输出;
- Qune3-4B投影尺寸;
- FP32 矩阵累积器;
- 第3天 - 第3天 - The day 3 SIMD 仍在用于 `M <= 8`.

## 从马特韦克到合作铺

香草单线点产品和单组8×8瓦是有用的
Metal 牵引控制,但没有提供足够的合作再利用。
多行键 prefill。将两者与 Python MLX 正确性预言 那个
性能表必须共享激活和减量化权重
一个更大的结果牌。

优化内核分配 4 SIMD 组合,或 128 线程到一个
32×32×32 瓷砖:

```plain
                  32 output columns
               +--------------------+
32 prompt rows |  four 16x16 SIMD   |
               |  output quadrants  |
               +--------------------+
                         ^
                         |
             shared 32-value K step
```

对于每32个减值步骤,线组:

1. 装入一个32x32 激活瓦片进入加固线组内存;
2. 将一个32×32重的瓦片拆开并分解;
3. 让四 SIMD 将两块瓷砖重新使用;
4. 累积4个16x16个四角体,来自 Metal 8×8 矩阵碎片;
5. 向下一个削减瓦垫垫垫。

40元素 共享- 记忆 踩板 32 值行以避免
无益的银行接入模式。 尾行和柱子为零填充或守卫
在最后的店铺。

 Metal 可使用内核 MLX低级钢铁 `BlockLoader` 和 `BlockMMA`
头作为构件。 这些帮助者提供合作负荷,
矩阵碎片簿记. 您的解决方案仍然拥有 W4A16 解包,
剥离、瓦片布局、原始、调度、分割政策和减少;
它不打电话 MLX数量化的 matmul 操作员。

## 任务1:保护工作量调度

修改 `QuantizedMatmul::eval_gpu` 输入
`src/extensions/src/quantized_matmul.cpp` 和
`quantized_matmul_simdgroup_w4a16_g128` 输入
`src/extensions/src/quantized_matmul.metal`保留第3天
`quantized_matvec_x4_fast_w4a16_g128` 函数完整 `M <= 8`.

保留第3天 decode 列表并添加后面的矩阵表
定量线性接口 :

```plain
M <= 8  -> quantized SIMD matvec
M > 8   -> 32x32x32 quantized SIMD-matrix kernel
```

通过累积显示新路径 `simd-matmul` 检查站。 测试
香草,瓷砖,和 MLX 对齐的形状和部分行和
栏牌。 结果必须保留模型造型 16 位 dtype.

## 任务2:使设备负载相近

继续修改 `quantized_matmul_simdgroup_w4a16_g128` (及其私人
Metal 帮助者,如果您考虑一个) in
`src/extensions/src/quantized_matmul.metal`; 不改变公众
`quantized_matmul` 装订。

使用一个合作区块加载器,这样相邻的线程和每个线程的局部
读作毗连交易。 这是时间表的要求,
不是化妆品细节 基准Q、K/V、大门/上下分别预测
在它们的 Quen3-4B 维度上, 输出网格的宽度和窄度都被覆盖。

## 任务3:同步量化参数

继续修改 `quantized_matmul_simdgroup_w4a16_g128` 输入
`src/extensions/src/quantized_matmul.metal`。此任务会改变平面
内核的负载/再使用策略,而不是其 C++ 或 Python 签名。

一个尺度和偏差适用于128的还原值。 每一次装入
32值的瓦片重复了4次相同的设备访问. 有一条线索负载
将32个输出列中的每个输出列的大小和偏差输入线组内存,
然后让那列的四条减重线线 重新用于下一个
4块减价瓦。

保持比例、偏差和无包装操作 BF16 存储, 而矩阵
积存器 FP32。在写最后的模型输出时一次播放。

## 任务 4: 仅需要工程逻辑

修改 `Qwen3ModelWeek2.__call__` 输入 `src/tiny_llm/qwen3_week2.py` 这样
`logits_to_keep=1` 词汇投影前切片. 不添加新内容
此模型级优化的扩展函数。

生成只需要最后的快速行来生成第一个样本 token.
接受 `logits_to_keep=1` 并只对该行应用词汇投影。
该基准将相同的最后记录工作量用于 MLX快速评分时
呼叫者仍然可以请求每个登录行.

## 任务 5: 验证、 基准和命名下一个瓶体

任务5没有添加函数 。 校验累积
`QuantizedMatmul::eval_gpu`/`quantized_matmul_simdgroup_w4a16_g128` 路径和
联合国 `Qwen3ModelWeek2.__call__` 任务1-4的投影边界。

```bash
pdm run build-ext
pdm run test --week 2 --day 6

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-decode-attention --variant week2-simd-matmul --variant mlx \
  --model qwen3-4b --input-len 128 --output-len 129 --warmup 2 \
  --prefill-logits last

pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-decode-attention --variant week2-simd-matmul --variant mlx \
  --model qwen3-4b --input-len 32 --output-len 33 --warmup 2 \
  --prefill-logits last
```

检查投影扫描以及完整的模型吞吐量. 继续
第七天,当长...`M` 预测是健康的,但短,窄的K/V
发射次数太少 32x32 结果瓦以填充 GPU如果相同
内核在逃仍然缓慢 `M`,改进它的负载或矩阵时间表
添加还原分区。

长久 `M`,二维的瓷砖网格已经很大. 不要强迫
下个优化: 额外的减少分区只会增加一个
临时缓冲器和另一次发射。

## 基准分析:查明未完成 Prefill 形状

在占用的控制形状和短 K/V 上比较矩阵内核
形状,然后在不启用 Split-K 的情况下对后者进行基准:

```bash
for context in 32 128 2048; do
  for projection in q k v o gate up down; do
    pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
      --section prefill-projections --context "${context}" \
      --prefill-projection "${projection}"
  done
done

```

发送公式给出未分割的 32 行 K 投影 32 独立
线组。

附上完整的模型 prefill 三角洲和每个预测表,32,128,
还有2 048行 不选择 Split- K 只因为预测仍然占据
多数 prefill首先需要长期或广泛的控制才能接近 MLX 时
短、窄的预测速度仍然过慢。

使用调度计算和短形状操作员扫描来确定
未分割的结果网格有太多的独立线组。 使用匹配的
长形状控制,以排除每块瓷砖内成本高昂的工作;如果它暴露了这种工作。
a 成本,修复第6天之前乘以电网. 那个
[参考检查站](./appendix-performance.md#day-6-use-cooperative-loads-for-quantized-prefill)
对齐 prefill 使用长短运算符控件和调度器增益
激发Split-K的几何. 剩下的算术热点会发出
你回到第6天时

> **选择性貌相证据.** 32/128行的归属可以证实
>形状分析,但不取代匹配的完整模型三角洲,
> 预测控制,以及上面的调度计算。

{{#include copyright.md}}
