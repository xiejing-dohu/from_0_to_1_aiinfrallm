# 🚧 第 2 周第 4 天：融合模型内核

> **现状:实验。** 见
> [第2周核查矩阵](./week2-overview.md#verification-status) (单位:千美元)
> 不断测试、在当地测量和仍在审查中的内容。

第3天消除了最大的预测缺口。 第4天的目标 RMSNorm, RoPE,以及
SwiGLU,它围绕这些预测在每一个变压器层发生. 第1个星期
表示为 Python `mlx.core` 方程式 方程式 第2周保留这些执行
请您写三篇 Metal 单独接口的内核 :

```plain
src/tiny_llm/week2_kernels.py
src/extensions/src/week2_kernels.cpp
src/extensions/src/week2_kernels.metal
```

您的解决方案仍然使用 MLX 数组及其扩展 API. MLX 时间表
图形节点,拥有其缓冲器,并发送 Metal 函数,但您的
解决方案拥有该函数内的算术。 您的解决方案不调用
`mx.fast.rms_norm`,
`mx.fast.rope`,或由MLX提供的SiLU执行.

> **选择性貌相证据.** 第3天内核组重播
> [参考-解决归属](./appendix-performance.md#checked-operator-attribution-that-selects-each-chapter)
> 显示优化预测背后的指针集群。 他们解释
> 章节顺序,但不是先决条件或接受门。

## 为什么融合有帮助

第1个星期 Python `mlx.core` 方程式已作为本地程序运行 GPU 内核
懒汉图. 重要的区别在于有多少操作和记忆
通过图表描述。

举例来说, RMSNorm 以 `mlx.core` 铸造、方块、
减少、取一个对等的平方根、乘法、再铸,并应用
学习体重。 编译器可以将一些相邻的元素逐项连接,但
线条缩小是一个边界。 中间值和多次发送
仍然有可能,因为,

单一引信 Metal 内核赋予你对整个操作器的明确控制:

- 一次发送取代了几次图表操作;
- 数值留在登记册或步骤之间的SIMD组存储中;
- 将浮积用于数字稳定性需要的地方;
- 在实际可行时阅读一次投入,只写最后的声调;
- 网格匹配点 decode 形状,而不是通用的变压器操作。

有用的比较不是 "Metal 对 Python 算术,但是一个
与几个通用内核的图表相对应。

## 任务1: RMSNorm

修改 `tiny_llm_ext::rms_norm`, `Week2RMSNorm::eval_cpu`,以及
`Week2RMSNorm::eval_gpu` 输入 `src/extensions/src/week2_kernels.cpp`,则
`week2_rms_norm` 函数在 `src/extensions/src/week2_kernels.metal`,以及
`FastRMSNorm.__call__` 输入 `src/tiny_llm/week2_kernels.py`启动头
装订, C++/Metal 文件, 而 CMake 注册已经存在
检查点; 替换失败关闭的机构而不是添加平行的API.

从一个开始 SIMD 每个输入行的组,然后作为基准。 隐藏了2 560个元素
行给出32个车道,每个车道大约80个序列元素;优化的内核发射 256
线程,或 8 SIMD 组合,每行。 每一组减少其部分
`simd_sum`; 车道 0 将 8 个部分总和写入线组内存; 第一个
SIMD 组进行第二次削减:

```plain
sum_sq = simd_sum(each lane's partial sum)
inverse_rms = rsqrt(sum_sq / hidden_size + epsilon)
output[i] = input[i] * inverse_rms * weight[i]
```

所有256条车道然后实现正常化,并拓宽其斜线。 这个引信
减少和产出通过一次发送,避免实现
平方声纳。 证明 bfloat16 所需的内核. 保留
减量、正常化和重量乘以浮标,然后投出
最后结果一次。 那个 Python 引用方程式回合一次,然后应用
重量,所以比较两者的耐受性,而不是期望比特-等同
结果。

那个 C++ 原始验证形状和 dtype,通过 MLX,
将缓冲器和伸缩常数捆绑起来,分配8个浮动部分金额,
每行发射256条线组。 将这一两级削减与
单 SIMD 组控制,以确定额外并行性是否抵消
目标机上的线组减小。

整合 `FastRMSNorm` 进入每两周规范,立即运行 RMSNorm
测试,并在写入前记录累积模型结果 RoPE:

```bash
pdm run build-ext
pdm run test --week 2 --day 4 -- -k rms
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint rmsnorm --model qwen3-4b
```

## 任务2: RoPE

修改 `tiny_llm_ext::rope`, `Week2RoPE::eval_cpu`,以及
`Week2RoPE::eval_gpu` 输入 `src/extensions/src/week2_kernels.cpp`,则
`week2_rope` 函数在 `src/extensions/src/week2_kernels.metal`,以及
`FastRoPE.__call__` 输入 `src/tiny_llm/week2_kernels.py`.

执行 RoPE 给模型的本地 `B, L, H, D` 设置。 一个天真的因素
内核为两个成员分别计算相同角度、正弦和余弦
每一对,每头一次。 相反,指定一个线程一对索引
和四个头的街区。 计算角度一次,然后旋转两个元素
那对四头

```plain
angle = (batch_offset + token_position) * base ** (-pair / (dims / 2))
real' = real * cos(angle) - imag * sin(angle)
imag' = imag * cos(angle) + real * sin(angle)
```

接受每批中每批中一个平面或一个平面 Python
包装 在发送前将这两个大小写规范为整数组 。 支助
每批在不同的请求中冲抵事项 decode 共用职位a
批号。

与构建位置数组、收集正弦值和余弦值的图表不同,
将头部拆分,执行若干项元素操作,以及
调和结果, 此内核读取每个输入对并写入每个
直接旋转元素 。 重用四头的三角形是关键
优化。 使用 Metal's `fast::exp2`, `fast::sin`,以及 `fast::cos` 联 合 国
BF16 路径。 将一个批量的抵消正常化 在一次的模型调用,
在层圈之外,而不是重建每层中相同的阵列。

替换 Python `mlx.core` RoPE 然后测试和测量
实施SwiGLU之前的累计检查站:

```bash
pdm run test --week 2 --day 4 -- -k rope
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint rope --model qwen3-4b
```

## 任务3:西格卢

修改 `tiny_llm_ext::swiglu`, `Week2SwiGLU::eval_cpu`,以及
`Week2SwiGLU::eval_gpu` 输入 `src/extensions/src/week2_kernels.cpp`,则
`week2_swiglu` 函数在 `src/extensions/src/week2_kernels.metal`,以及
`swiglu` 输入 `src/tiny_llm/week2_kernels.py`.

SwiGLU 将大门和上方的树枝结合在一起:

```plain
output = (gate / (1 + exp(-gate))) * up
```

每个元素执行为一条线程 。 线程载荷 `gate` 和 `up`,
用一个指数值来评价SiLU,将分支乘以乘,并执行一个
输出写入 。 第1周的表格比较容易检查,但它描述了 `abs`,
`exp`,除法、选择和乘法作为单独的数组操作。
熔化的内核去除这些中间伸缩器和发送边界.

立即整合引信表达式,并记录第三个检查站:

```bash
pdm run test --week 2 --day 4 -- -k swiglu
pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint swiglu --model qwen3-4b
```

## 任务4:验证累积模式

校验累积开关 `Qwen3ModelWeek2.__init__` 和呼叫站点
输入 `Qwen3MultiHeadAttention.__call__` 和 `Qwen3MLP.__call__`任务4不应
引入另一个扩展函数;它从
任务1-3.

打开三个内核后 C++ MLX 原始人,运行完整
测试文件以验证其组成。 保留 `qwen3_week1.py` 第1周举行
Python 运算符,并让"周2"接口通过"周3"服务模式重新使用.

```bash
pdm run build-ext
pdm run test --week 2 --day 4
```

比较 Python 带有容忍度而不是比特值的参考方程式
平等无边. 测试 RoPE 与斯卡尔和每批相抵。 总是打电话 `mx.eval`
在测量这些懒惰操作时, 时间迭代。

操作者基准也必须比较相同的逻辑 RoPE 设置。 
RoPE 内核接受模型本地化 `B, L, H, D` 声纳。 `mx.fast.rope`
预期 `B, H, L, D`,所以在 MLX 电话和
后转其结果. 没有那些转音,一个一键
基准不慎将头轴作为顺序位置和
计时不再计量等效操作。

## 基准分析:决定 软骨是否停留

把三个累计的检查站分开 以免倒退隐藏起来
在其综合收益中:

```bash
pdm run bench-week2-progression --offline --solution tiny_llm --repeats 4 \
  --variant week2-quantized-matvec \
  --variant week2-rmsnorm --variant week2-rope --variant week2-swiglu \
  --variant mlx --model qwen3-4b \
  --input-len 128 --output-len 129 --warmup 2 --prefill-logits last

pdm run bench-week2-operators --solution tiny_llm --model qwen3-4b \
  --section model-kernels --context 128
```

附加每个累积模型行, 三个 Python 参考/优化/MLX 运算符
排队 直接发送追踪 让相应的基准结果决定
内核是否保留。

继续到第5天 当所有三个正确的门通过, 直接
引信发送源追踪到达预定内核,累积行
保持收益,三个操作员的比较证明保持保险丝是合理的
执行。 第5天,然后测试注意力是否是下一个可移动缺口
通过在设置调度守护器之前先扫描缓存上下文和查询长度。

> **选择性貌相证据.** 那个
> [参考检查站](./appendix-performance.md#day-4-fused-model-kernels)
> 将累积测量和操作员测量成对并更新
> 归属。 这种归属可以解释过渡,但不能
> 替换上述检查站证据。

{{#include copyright.md}}
