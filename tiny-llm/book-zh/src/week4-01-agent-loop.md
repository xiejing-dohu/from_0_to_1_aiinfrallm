# 第 1 天：从生成到智能体循环

> 🚧 **早期审查 WIP:** 本章公开供早期审查,并可
> 变化。 运行代理或允许写入时使用可支配的工作空间
>或命令。

第1周的解码器产生一次文本并退出. 代理多次转弯
文本输入动作,执行该动作,并将结果反馈给
型号。 今天你们要明确控制流

> **执行情况:** 当前第1天的学习者检查覆盖
> `initial_messages()`:它保存系统指令和任务并拒绝
> 一个空任务。 完整 `run_agent()` 循环在本章后面描述
> 在参考基线中存在,但其重点检查目前位于
> 第6天。 本章其余部分作为第1天的扩展,而不是作为
>第1天测试证明的行为.

## 当前仓库检查点

执行 `initial_messages(task, system_prompt)` 输入
`src/tiny_llm/agent/generation.py`,然后运行 :

```bash
pdm run test --week 4 --day 1
```

预期结果是,第1天的两次测试都通过了,而没有加载一个模型。 改为
检查所提供的执行, 运行
`pdm run test-refsol --week 4 --day 1`.

## 学习目标

到头来,你就可以:

- 解释模型、代理循环和工具之间的区别;
- 将工具呼叫和最后答复作为结构化行动;
- 在谈话中保留助理动作和工具观察;
- 完成后可靠地停止生产、产出或分级预算。

## 动作, 不自由格式命令

从JSON协议开始 因为它在每一个跟踪中都能看到 并且与
不显示本地工具调用模式。 一个助手转弯 正好产生
两个形状中的一个:

```json
{"tool":"read_file","path":"README.md"}
```

```json
{"final":"The project implements a small Qwen3 inference stack."}
```

解析JSON只是第一次检查. 解码值必须是对象, 必须
完全包含其中之一 `tool` 或 `final`,并且必须包含
动作的阴谋。 拒绝后面的文本,而不是默默地忽略它。

返回验证失败到模型作为观测. 这让模型
在不隐藏失败的情况下修复一个错误的动作 :

```text
error: missing fields for read_file: path
```

## 循环

与推理和工具执行保持协调:

```python
def run_agent(task, generate, workspace, limits):
    messages = initial_messages(task, build_system_prompt(workspace))
    events = []
    for step in range(1, limits.max_steps + 1):
        response = generate(messages)

        try:
            action = parse_action(response, workspace.available_tools)
        except AgentError as error:
            result = f"error: {error}"
            events.append(AgentEvent(step, response, None, result))
            messages = append_tool_result(messages, response, result)
            continue
        if isinstance(action, FinalAction):
            events.append(AgentEvent(step, response, action, None))
            break

        result = workspace.execute(action)
        events.append(AgentEvent(step, response, action, result))
        messages = append_tool_result(messages, response, result)

    # Construct AgentRun with either "completed" or "step_limit".
```

该循环拥有预算和停止条件等政策。 模型适配器拥有
符号化和解码。 工具登记册拥有计划和执行。 这些
当届会和取消于本周晚些时候到达时,边框将很重要。

## 保存路径

现在,一个记忆清单就足够了。 当前情况 `AgentEvent` 记录
步骤、原始反应、分析有效时的动作和结果。 目标追踪应
最终也使得这些运行级别的事实可以发现:

- 用户的任务;
- 助理的原始反应;
- 解析的动作,如果有效;
- 工具结果或验证错误;以及
- token 计算和计算可用时间。

不要只存储最新的提示字符串 。 命名的活动比较容易检查
并可以在以后不逆向工程化聊天模板的情况下被序列化.

## 计划循环演习

保留 `generate_response()` 负责一个模型反应
`run_agent()` 拥有循环。

在不装入模型的情况下实施和测试这些案例:

1. 执行有效的工具行动并附上其结果。
2. 最后行动在不执行工具的情况下阻止循环。
3. 无效的JSON成为有用的观察,循环继续.
4. 一个未知工具在执行前被拒绝。
5. 循环停止之后 `max_steps` 即使模型从未完成。

使用返回预设字符串的假模型和假工具注册
记录电话。 多数代理- loop 行为是普通的确定码, 并且确实
不需要昂贵的模型测试。

## 检查站

计划循环切片实施后,应当进行以下跟踪.
可能为:

```text
user      inspect the repository
assistant {"tool":"read_file","path":"README.md"}
tool      # Tiny LLM ...
assistant {"final":"This repository teaches LLM inference and serving."}
```

这个工具仍然是这个计划中的片段的支点. 当前源树
限定的工作空间 API 由第2天和第3天的检查站和
当前循环行为在第6天前得到验证.

{{#include copyright.md}}
