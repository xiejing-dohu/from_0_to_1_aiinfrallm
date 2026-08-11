# 第 4 天：交互式会话与恢复

> 🚧 **早期审查 WIP:** 本章公开供早期审查,并可
> 变化。 运行代理或允许写入时使用可支配的工作空间
>或命令。

从头三天开始的无国籍循环,每当
进程退出。 本章使事件流持久并添加可重复使用
生成缓存时,不做任何一种表示都取决于另一种。

> **执行情况:** 参考执行,学习者 API 表面,
> 本章中的重点测试可以执行。 这个章子还是WIP
> 尽管检查站是可以执行的。

## 检查章节

在下方执行会话和生成 API `src/tiny_llm/agent/`,然后运行 :

```bash
pdm run test --week 4 --day 4
```

使用 `pdm run test-refsol --week 4 --day 4` 检查所提供的执行情况。
测试使用临时目录和假缓存; 不执行模型
命令或破坏性子进程。

## 三国

实施工作有意将以下内容分开:

1. **持久性会话状态 :** 只附加用户、助手、工具和生命周期
   事件。
2. **模式可见背景:** 从这些事件中重建的语义聊天信息.
3. **KV 状态 :** 实现进程优化 token 前缀。

JSONL日志为犬科. 丢失或关闭a KV cache 只有下一个
转身显示寒冷 prefill这并不能使谈话变得无法进行
续会。

## 仅追加会话日志

`SessionEvent` 存储 ID、 UTC 时间戳、 事件类型、 可选父标识,以及
类型特定数据。 `SessionLog.append()` 序列化 ID 任务和附件,
冲洗线条,和呼叫 `fsync` 返回之前, 持续开会开始
含有已解析工作空间、模型/后端/模板的元数据
标识符,并装入项目指令。

代理记录的事件顺序如下:

```text
user_message
assistant_message        # durable before parsing or executing
tool_call
tool_result
run_finished
```

如果一个进程在之后退出 `tool_call`,继续附加一个中断的简洁
`tool_result`。它从不重复副作用或创造成功的结果。
格式不正确、大小过大、重复ID、连线或元数据不一致的日志
失败关闭 。

会话记录可以包含源文本和命令输出. 他们住在地下
`.tiny-llm/sessions`被基特忽略 并隐藏在模型的
工作空间工具。 将该目录视为敏感的本地数据.

## 继续边界

`SessionStore` 创建、加载和选择最新的会话。 装入验证
已解析的工作空间和模型/后端/模板标识符。 会话不能
静静地重放在不同的寄存器或推理模板上.

在创建和恢复时,商店读取工作空间根 `AGENTS.md` 当它
是一个有边框的普通 UTF-8 文件。 该快照和SHA-256摘要被录制下来.
如果文件更改,则 `instructions_changed` 事件解释哪个策略是
现在可见。 递归指令发现和任意配置文件
故意推迟。

CLI支持:

```text
agent TASK                 start and persist a new session
agent --interactive TASK   accept follow-up messages in the same process
agent --continue           resume the newest session for this workspace/model
agent --session ID         resume one selected session
agent --no-session TASK    run without creating a session file
```

`--continue` 和 `--session` 是相互排斥的。 `--no-session` 无法是
与复旗相结合。 现行第3天安全规则仍然适用于每一天
恢复后的工具呼叫。 完成的会议需要新的互动后续行动;
永远不要创建非主动模式转弯。

## 重建背景

上下文构建器将用户和助理事件映射到聊天角色和
将已完成的工具结果包装为用户观察。 寿命周期、时间和
元数据事件仍然是只审计的。 寻回的无与伦比的呼叫变成了正常的
错误观察,所以下一个模型转弯可以选择安全的下一个动作.

第4天,一旦谈话,仍然使用字符限制保留助手
正在重建。 减少记忆和结构化摘要属于第5天;
耐久的原木从未修剪过。

## 重用 KV 状态

`GenerationSession` 保留现有的一个参数 `Generate` 边框 :

```python
response = generation_session(messages)
```

它会制作和标记所请求的信件,比较 token 身份证
以每个活缓存层为代表的前缀,并计算其最长的共性
前缀。 在倒转前,它验证所有层的抵消都符合它自己的
记账 有效的不同后缀是每层重音,只有新的
后缀为前缀。 任何异议、 不支持的倒带或缓存错误
释放整个缓存集并开始冷。

`GenerationStats` 报告输入、重用、重用、预选和产出 token
加上转弯是否开始冷。 具有代表性的不变量是:

```text
every cache offset == len(the token IDs represented by the live cache)
```

最后样本 token 只有在实际通过
那个模特儿 `close()` 具有同位素并释放每个缓存。 温暖和冷路
必须产生同样的贪婪的反应。

可重复使用的缓存在课程周2/3模型适配器中执行. 那个
`--solution mlx` 兼容后端保留其无国籍生成器; 耐用
会话重播仍然在那里工作。

## 锻炼

1. 通过JSONL进行每一场会话的圆路。
2. 核实在工具运行前助理反应是否持久。
3. 恢复未匹配的工具呼叫,而不重新执行。
4. 拒绝工作空间或模型不匹配。
5. 变化 `AGENTS.md` 并检查记录的过渡。
6. 确认 `--no-session` 写什么。
7. 用假层比较缓存再利用、倒带和冷落回计数器。
8. 关闭两次实时生成会话而不泄露缓存。

第5天增加了已编制的预算和结构化的压缩。 增加第6天
突变恢复、检查站、取消、引导和分支。 磁盘 KV
快照和多进程会话写入器不是此检查点的一部分 。

{{#include copyright.md}}
