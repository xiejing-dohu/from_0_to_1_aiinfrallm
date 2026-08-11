# 第 5 天：面向 Token 的上下文压缩

> 🚧 **早期审查 WIP:** 本章公开供早期审查,并可
> 变化。 运行代理或允许写入时使用可支配的工作空间
>或命令。

只有附件的会话会永远增长,但模型有一个有限的上下文窗口.
第5天获得一个没有删除或改写的有界限的模型可见工作集
从第4天开始的持久痕迹。

> **执行情况:** 参考执行,学习者 API 表面,
> 本章中的重点测试可以执行。 这个章子还是WIP
> 尽管检查站是可以执行的。

## 检查章节

执行上下文 API `src/tiny_llm/agent/`,然后运行 :

```bash
pdm run test --week 4 --day 5
```

使用 `pdm run test-refsol --week 4 --day 5` 用于提供的执行。 那个
使用合成记录和注入编码器/摘要器进行新的压缩测试;
第3天保留的安全测试使用模拟程序而不是执行代理
命令。

## 可执行政策

`ContextPolicy` 明确预算:

```python
ContextPolicy(
    max_tokens=32_768,
    reserve_tokens=8_192,
    summary_max_tokens=1_024,
    max_tool_result_tokens=4_096,
    min_recent_turns=2,
)
```

工作输入限制是 `max_tokens - reserve_tokens`,或24,576个
课程 Qwen3-4B 默认值。 储备包括下一次反应和观察;
这不是额外的模型上下文。

`ContextManager` 接收到生成时使用的相同的精确信件编码器 :

```python
manager = ContextManager(generation.encode_messages, policy)
window = manager.prepare(session, system_prompt, summarize)
```

`ContextWindow.token_ids` 因此,将提出的全部请求计算在内,包括:
聊天模式框架、系统工具计划、项目指示、信息、
概要和可见的工具结果。 字符计数和每个消息 token
估计数不作为限额。

如果无法固定的锚 加上最近最小的尾巴 准备
提高 `ContextLimitError`循环停止一次 `reason="context_limit"`;
它从不放弃当前请求、 反复重试压缩或调用
带有已知的流转请求的模型。

## 总结前的束线观测

持久 `tool_result` 事件总是保留其原有的限定工具输出。
只有其模型可见的渲染才能进一步减少:

- 清单保留有用的头;
- 命令式输出保持有用的尾部;
- 类似文件的文本用遗漏标记将头和尾分开。

管理者使用精确的编码器来验证所降低的结果
`max_tool_result_tokens`。这阻止了一个大型观测数据消耗
整个压缩预算,同时保存用于审计和
后评.

## 结构化工作摘要

旧的完整事件在模式可见的背景中替换为严格的
`WorkingSummary`:

```json
{
  "goal": "Fix parsing of empty configuration values",
  "constraints": ["Do not change the public configuration schema"],
  "facts": ["parse_value is defined in src/config.py"],
  "changed_files": ["src/config.py"],
  "validation": ["test_empty_value still fails"],
  "failed_approaches": ["The first exact edit produced a tool error"],
  "next_step": "Inspect normalization before parse_value"
}
```

解析器需要这些密钥, 非空格所需的字符串, 不可改变
字符串拖曳, 以及固定项/ 大小边框 。 额外的密钥和错误的类型
拒绝。 最初的目标仍然是固定的。 即使是一个被接受的模型
汇总、更改路径和命令状态与成功调和
结构化工具结果,而不是嵌入在不信任输出文本中的债权。

一个可选的总结会收到一个专门的计划指令和语义
消息并可能返回 JSON 一次。 管理者要求的精确编码
,并在调用模型前保留已配置的汇总输出。 生下来的
答复或错误只记录在有约束的审计中 `summary_attempt` 活动。
无效的 JSON, 计划失败, 例外, 或超预算请求或摘要
立即选择决定性的倒置; 没有递归式重试 。

CLI为摘要工作提供新的临时代源缓存,所以
主代理缓存在验证的压缩事件之前保持不变
持久。 测试和轻量级集成可能会省略回调和选择
直接决定战略。

## 持久压缩标记

压缩附加此事件而不是更改旧事件 :

```json
{
  "covered_through_event_id": "...",
  "strategy": "model",
  "fallback_reason": null,
  "summary": {"goal": "..."},
  "input_tokens_before": 25001,
  "input_tokens_after": 3812
}
```

覆盖边界必须参照现有事件,不能跨越
未匹配的工具调用 。 从最新的结构开始重复压缩
概要加上以后的事件;只有最新的摘要是模式可见的。 原文
JSONL事件仍然可以检查和重播。

`ContextWindow` 返回准确 token ID,可见工具结果字节数,
这个准备是否附着一个紧凑, 以及事件的身份。 那个
辅助事件记录这些上下文参数。 课程 `GenerationSession`
后端也记录 `GenerationStats`——其第5天的栏目现在包括休养;
无国籍人 MLX 兼容性后端留下生成度量未知.

## 缓存调节

压缩会改变旧部分的速报 。 循环通过
给第4天的新语义信息 `GenerationSession`,使用同样的 token
最长的常见前缀路径与其它转弯一样 。 每层都先经过验证
倒带; 兼容层倒带不同的后缀 prefill 新书
简表/尾注,而任何不一致的状态都会因寒冷而丢弃 prefill.

事件日志和摘要为语义状态. K/V 口径仍为衍生状态
,从不进行总结、编辑,也不需要重新启动正确性。

## 锻炼

1. 构建一个包含重复阅读、编辑和验证的合成会话
   观察。
2. 衡量其充分提供的情况 token 计数。
3. 核实预算不足编制不写压缩事件。
4. 压缩旧前缀并重新装入JSONL会话.
5. 强迫无效摘要JSON并检查确定性的后退标记。
6. 确认原始工具结果保持不变。
7. 调和温暖的一代缓存并将其反应与冷跑相比较。
8. 将预算降低到锚点以下,并观察一次故障停机。

第6天增加了写头突变恢复,检查站,取消,取消,
方向和树枝 RAG、磁盘K/V快照、适应性缓存休眠药和
第7天的分级仍然明确延长。

{{#include copyright.md}}
