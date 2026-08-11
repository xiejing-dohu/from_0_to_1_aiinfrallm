# 配置开发环境

要遵循这个方向,你需要苹果硅的Mac. 该项目利用方案支助和项目管理进行依赖和环境管理。

## 安装 PDM

按照[官方安装指南](https://pdm-project.org/en/latest/)安装 PDM。

## 仓库克隆

```bash
git clone https://github.com/skyzh/tiny-llm
```

资料库的结构如下:

```
src/tiny_llm/ -- your implementation
src/tiny_llm_ref/ -- the reference implementation
tests/ -- unit tests for your implementation
tests_refsol/ -- unit tests for the reference implementation
book/ -- the book source
```

如果课程中遇到问题,可提供参考执行。

## 安装依赖关系

```bash
cd tiny-llm
# This creates a virtual environment and installs all dependencies.
pdm install -v
```

## 检查安装

```bash
pdm run check-installation
# The reference solution should pass all Week 1 tests.
pdm run test-refsol -- -- -k week_1
```

## 运行单元测试

你的密码在 `src/tiny_llm`。您可以使用:

```bash
pdm run test
```

## 下载模型参数

我们用正式的4位 Qwen3 MLX 模型文件。 默认模式是 `Qwen/Qwen3-0.6B-MLX-4bit`,它足够小
被贬低者 Python 第1周的执行情况。 如果设备有较多的内存,也可以尝试更大的 Qwen3 模特儿们

遵循 [Higging Face CLI 引导](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) 以安装 `hf`
命令行工具。

模型参数由Hugging Face主持. 在用您的证书认证 CLI 后, 用 :

```bash
hf auth login
hf download Qwen/Qwen3-0.6B-MLX-4bit
# Optional larger models:
hf download Qwen/Qwen3-1.7B-MLX-4bit
hf download Qwen/Qwen3-4B-MLX-4bit
```

然后,你可以运行:

```bash
pdm run main --solution ref --loader week1
```

命令应当加载参考模型并打印生成的文本.

在第二个星期,我们会写 C++ 和 Metal 内核. 第1周结束时将讨论所需的其他工具。

{{#include copyright.md}}
