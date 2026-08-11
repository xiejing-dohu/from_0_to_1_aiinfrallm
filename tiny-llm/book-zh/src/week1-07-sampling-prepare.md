# 第 1 周第 7 天：采样与第 2 周准备

第7天,我们将实施若干抽样战略,并为第2周的发展环境做好准备。

## 任务1:取样

第六天,我们实施了贪婪的解码。 在这个任务中,我们将加入温度,顶-k,和顶-p(核)采样.

```
src/tiny_llm/sampler.py
```

- [mlx-lm采样器执行](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py)

### 温度取样

何时 `temp=0`使用贪婪的解码。 何时 `temp` 大于 0, 选择下一个 token 从日志概率分布。
温度越高,分布就越平坦,使得概率越低的标志更有可能出现,输出种类也就越多.

为了实施温度取样,将日志概率除以温度并将其传递给
`mx.random.categorical`.

```bash
pdm run main --solution tiny_llm --loader week1 --model qwen3-0.6b --sampler-temp 0.5
```

### 顶端取样

顶级采样只保留 `k` 日志概率最高的标志。 在温度缩放前应用此过滤器 。

使用 `mx.argpartition` 在顶部之外查找索引 `k`,用 `-mx.inf`,然后应用
温度取样。

```bash
pdm run main --solution tiny_llm --loader week1 --model qwen3-0.6b --sampler-temp 0.5 --sampler-top-k 10
```

### 顶( 核) 取样

顶级采样保留了累积概率达到或超过的最小的高概率标志集. `p`.
在温度缩放前应用此过滤器 。

一个执行用途 `mx.argsort` 命令日志概率从最高到最低,适用 `exp` 恢复
概率,适用 `cumsum` 计算累积概率。 保持一个 token 之前的累积概率
这还不到 `p`;这包括 token 越过门槛。 将剩余的日志概率用
`-mx.inf`,然后应用温度取样。

```bash
pdm run main --solution tiny_llm --loader week1 --model qwen3-0.6b --sampler-temp 0.5 --sampler-top-p 0.9
```

## 任务2:为第2周做准备

在第二个星期,我们将优化 Qwen3 为基础设施提供服务 C++ 和 Metal 内核. 您需要 Xcode 及其密码
命令行工具,包括 Metal 编译器,用于构建它们.

1. **安装 Xcode :**

    从 Mac App Store 或 [苹果开发者网站] 安装 Xcode](https://developer.apple.com/xcode/) (这可能需要一个苹果开发者账户).

2. **启动 Xcode 并安装组件 :**

    安装后,至少发射一次Xcode. 它可能会促使您安装额外的 macOS 组件; 请这样做( 这通常是默认选项) 。

3. **安装 Xcode 命令行工具 :**

    打开终端并运行 :

    ```bash
    xcode-select --install
    ```

4. **设置默认 Xcode 路径( 如果需要) :**

    确保您的命令行工具指向您新安装的 Xcode 。 你可以通过运行来做到这一点:

    ```bash
    sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
    xcode-select --print-path
    ```

    如果 Xcode 在别处安装, 请调整路径 。

5. **接受 Xcode 许可证 :**

    您可能需要接受 Xcode 许可证 :

    ```bash
    sudo xcodebuild -license accept
    ```

6. **校验 Metal 编译器 :**

    ```bash
    xcrun metal --version
    ```

    X码26 Metal 工具链可能是一个单独的组件。 如果命令报告缺失 Metal 工具链,
    下载并再次验证编译器 :

    ```bash
    xcodebuild -downloadComponent MetalToolchain
    xcrun metal --version
    ```

7. **安装 CMake :**

    ```bash
    brew install cmake
    cmake --version
    ```

(本指示由刘 Jin义慷慨提供)](https://github.com/KKKZOZ).)

通过将代码编译到 `src/extensions`,包含 `axpby` 函数从
公务 MLX 扩展教程 :

```bash
pdm run build-ext
pdm run build-ext-test
```

它应该打印 `correct: True`.
其他导出扩展名是贴有标签的失败关闭启动器根
第2周或第3周执行这些检查点;这一设置检查呼叫
仅限 `axpby`.

如果你是新来的 C++ 或 Metal,在继续前尝试一些小练习。 例如,采用元素操作
例如, `exp`, `sin`,以及 `cos`,然后用它们代替相应的 MLX 您模式中的操作
执行。

结束了第1周。 我们实施了所有必要的组成部分 Qwen3在第二周,我们将优化
为苹果硅服务基础设施.

{{#include copyright.md}}
