# from_0_to_1_aiinfrallm

📖 **大模型 AI 推理基础设施与 CUDA 高性能编程 — 个人学习笔记与优质网络资源中文整理**

本仓库用于记录个人在 **LLM 大模型推理服务系统（Inference Engine）**、**AI 基础设施（AI Infra）** 以及 **CUDA/Triton 算子优化** 领域的学习历程。主要包含了对网络优质英文技术博客、学术论文及开源项目的翻译、精读梳理与系统化笔记。

---

## 📚 目录指南与资源来源

| 章节文档 | 核心主题 | 学习笔记概述与参考源 |
| :--- | :--- | :--- |
| 📄 [1. KV Cache 管理架构演进](1%20_kccahe管理架构演进.md) | 内存管理 / PagedAttention | 梳理 KV Cache 显存碎片痛点与 PagedAttention 虚拟内存分页思想原理 |
| 📄 [2. 静态批处理与动态批处理](2_大模型推理服务_%20静态批处理和动态批处理.md) | 调度策略 / Continuous Batching | 整理 Static Batching 与 Continuous Batching (Iteration-level) 动态调度对比 |
| 📄 [3. PD 分离架构深度解析](3_大模型推理服务_PD分离架构.md) | 异构计算 / PD 拆分 | 记录 Prefill（计算密集型）与 Decode（访存密集型）节点解耦及集群扩展架构 |
| 📄 [4. vLLM 核心技术深度解析](4_vLLM高吞吐大模型推理系统架构与核心技术深度解析.md) | 推理引擎 / 系统设计 | 深入剖析 vLLM 整体设计、Prefix Caching、Chunked Prefill 与 Guided Decoding |
| 📄 [5. Inside vLLM：系统剖析与源码图解](5_Inside_vLLM_Anatomy_of_a_High_Throughput_LLM_Inference_System.md) | 核心原理 / 经典讲义翻译 | 精读翻译自 Aleksa Gordić 深度好文 *Inside vLLM: Anatomy of a High-Throughput LLM Inference System*，详述 V1 架构与生命周期 |
| 📄 [6. CUDA 高性能编程与算子优化教程](6_CUDATutorial_CUDA高性能编程与大模型推理教程.md) | 算子调优 / CUDA & Triton | 对开源项目 [PaddleJitLab/CUDATutorial](https://github.com/PaddleJitLab/CUDATutorial) 进行框架梳理，包含 Reduce/GEMM/Conv 级联优化 |
| 📄 [7. TensorRT-LLM 快速入门与架构解析](7_TensorRT_LLM快速入门指南与架构解析.md) | 推理引擎 / NVIDIA 优化 | 总结 NVIDIA 官方 [TensorRT-LLM Quick Start Guide](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)，涵盖 AOT 编译、`trtllm-build`/`trtllm-serve` 与性能特性 |
| 📄 [8. TensorRT-LLM 完整实战博客与算子适配指南](8_TensorRT_LLM完整实战博客与算子适配指南.md) | 算子适配 / 链路优化面试干货 | 深度拆解 NVIDIA 与阿里云 8 篇 TensorRT-LLM 官方实战博客，汇总 Qwen 适配、W4A16/FP8 量化、Plugin 开销陷阱与 10 大硬核面试模板 |
| 📄 [9. 阿里 RTP-LLM 工业级推理引擎架构与生产实践](9_阿里RTP_LLM工业级推理引擎架构与生产实践.md) | 工业级推理 / OOM & 负载均衡 | 深入剖析阿里开源 [RTP-LLM](https://github.com/alibaba/rtp-llm)，解析 PagedAttention、Radix Tree 前缀复用、OOM 水线防御与集群负载均衡 |
| 📄 [10. 从零手写实现 LLM 的 KV Cache 机制](10_Coding_the_KV_Cache_in_LLMs_From_Scratch.md) | 代码实现 / PyTorch 手写 | 精读 Sebastian Raschka 博士讲义 [Coding the KV Cache from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)，手写带 KV Cache 的 Multi-Head Attention，分析预分配与滑动窗口优化 |

---

## 🛠 学习涉及技术栈

* **推理框架**: vLLM, TensorRT-LLM, RTP-LLM, SGLang, TGI
* **内存与调度**: PagedAttention, Continuous Batching, Chunked Prefill, Prefix Caching, PD Separation
* **算子与加速**: CUDA C++, OpenAI Triton, FlashAttention, CUTLASS, CUDA Graph

---

## 声明与致谢

本仓库内容均为**个人学习目的**整理与中文翻译笔记。文中引用的图文与算法思想版权归原作者及开源社区所有（包括但不限于 vLLM Team, NVIDIA TensorRT-LLM, 阿里巴巴 RTP-LLM, Sebastian Raschka 博士, Aleksa Gordić, PaddleJitLab, CodeFuse, 阿里云, Orca, DeepSeek 等）。感谢前沿开发者们的无私分享！
