"""s05: 在 Hook 版 Agent 上加入 TodoListMiddleware。

这个文件的主线是：
1. 读取 .env 里的模型配置。
2. 定义 Hook 系统，监听用户提交、工具调用前后、Agent 结束。
3. 定义本地 shell / 文件读写 / glob 工具，并在调用前做权限检查。
4. 通过 TodoListMiddleware 给 Agent 增加 write_todos 工具。
5. Agent 执行结束后打印 todo 列表和本轮新增消息。
"""

# Callable 用来给 handler 这类“可调用对象”做类型标注。
from collections.abc import Callable
from typing import Any
from dotenv import load_dotenv
# override=True 表示 .env 里的变量可以覆盖系统环境变量里的同名值。
load_dotenv(override=True)

# 这些对象来自 LangChain Agent 的 middleware 机制。
# TodoListMiddleware 会额外提供 write_todos 工具，让 Agent 在复杂任务里维护待办列表。
from langchain.agents.middleware import AgentState,TodoListMiddleware, before_agent, after_agent, wrap_tool_call
# ToolCallRequest 表示一次工具调用请求，里面包含工具名、参数、调用 id 等信息。
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
# Runtime 是 LangGraph 的运行时对象，这里主要用于 middleware 函数的类型标注。
from langgraph.runtime import Runtime
# Command 是 LangGraph 控制图执行流的返回类型之一，这里用于工具 Hook 的类型标注。
from langgraph.types import Command
from pathlib import Path
from langchain_core.tools import tool
import os, subprocess
# AIMessage 是模型回复消息的类型，print_assistant_message 会用它做类型标注。
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

HOOKS = {'UserPromptSubmit': [], 'PreToolUse': [], 'PostToolUse': [], 'Stop': []}

def register_hook(event: str, callback):
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None

# before_agent 会在 Agent 正式运行前触发。
@before_agent
def user_prompt_submit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    message = state.get('messages', [])
    if message:
        last = message[-1]
        content = last.get('content') if isinstance(last, dict) else getattr(last, 'content', None)
        trigger_hooks('UserPromptSubmit', content)
    return None

# wrap_tool_call 会包住每一次工具调用，适合做权限检查和日志打印。
@wrap_tool_call
def tool_hook(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
    tool_name = request.tool_call['name']
    tool_args = request.tool_call.get('args', {})
    blockd = trigger_hooks('PreToolUse', tool_name, tool_args)
    if blockd:
        return ToolMessage(content=str(blockd), tool_call_id=request.tool_call['id'], name=tool_name, status='error')
    result = handler(request)
    trigger_hooks('PostToolUse', tool_name, tool_args, result)
    return result

# after_agent 会在 Agent 本轮执行结束后触发。
@after_agent
def stop_hook(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    trigger_hooks('Stop', state.get('messages', []))
    return None


WORKDIR = Path.cwd()
MODEL_ID = os.getenv('MODEL_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SYSTEM = f'you are a coding agent at {WORKDIR}. Use tools to solve tasks. Act dont explain'
OPENAI_BASE_URL = os.getenv('BASE_URL')
dangerous = ['rm -rf /', 'sudo', 'shutdown', 'reboot', 'mkfs', 'dd if=', '> /dev/sda']

def check_deny_list(command: str) -> str | None:
    for pattern in dangerous:
        if pattern in command:
            return f'blocked:{pattern} is on the deny list'
    return None

def resolve_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (WORKDIR / candidate).resolve()


def check_rules(tool_name: str, args: dict) -> str | None:
    if tool_name == 'run_bash':
        command = args.get('command', '')
        if command.strip().lower().startswith("del ") or any((kw in command for kw in ['rm ', '> /etc/', 'chmod 777'])):
            return 'potentially destructive command'
    if tool_name in ('run_write', 'run_edit', 'run_read'):
        path = args.get('path', '')
        if not resolve_path(path).is_relative_to(WORKDIR):
            return 'Working outside workspace'
    return None

def ask_user(tool_name: str, args: dict, reason: str) -> bool:
    print(f'\nWarning: {reason}')
    print(f'Tool: {tool_name}({args})')
    choice = input('Allow? [y/N] ').strip().lower()
    return choice in ('y', 'yes')

def check_permission(tool_name: str, args: dict) -> bool:
    if tool_name == 'run_bash':
        reason = check_deny_list(args.get('command', ''))
        if reason:
            print(f'\nBlocked:{reason}')
            return False
    reason = check_rules(tool_name, args)
    if reason:
        return ask_user(tool_name, args, reason)
    return True

# 下面四个函数是具体 Hook 回调：打印用户输入、工具调用、工具结果和结束事件。
def on_user_prompt_submit(content):
    print('[UserPromptSubmit]', content)

def on_pre_tool_use(tool_name, tool_args):
    print('[PreToolUse]', tool_name, tool_args)
    if not check_permission(tool_name, tool_args):
        return 'Permission denied'

def on_post_tool_use(tool_name, tool_args, result):
    print('[PostToolUse]', tool_name, tool_args)
    print('result:', getattr(result, 'content', result))

def on_stop(messages):
    print('[Stop]', len(messages))
# 把上面定义的回调注册到 HOOKS 表里，Agent 执行时才会真正触发。
register_hook('UserPromptSubmit', on_user_prompt_submit)
register_hook('PreToolUse', on_pre_tool_use)
register_hook('PostToolUse', on_post_tool_use)
register_hook('Stop', on_stop)

# run_bash 是 shell 工具：执行命令并把 stdout/stderr 返回给模型。
@tool
# 安全边界：shell=True 仅为教学演示，黑名单/路径检查不等于安全边界；生产请使用权限中间件 + 沙箱。
def run_bash(command: str) -> str:
    """Execute a shell command in the current workspace."""
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else '(no output)'
    except subprocess.TimeoutExpired:
        return 'Error: Timeout(120s)'
    except OSError as e:
        return f'Error: {e}'

# run_read 是读文件工具，支持用 limit 限制返回的行数。
@tool
def run_read(path: str, limit: int | None=None) -> str:
    """Read a UTF-8 text file, optionally limiting the returned line count."""
    try:
        lines = resolve_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f'...({len(lines) - limit} more lines)']
        return '\n'.join(lines)
    except Exception as e:
        return f'Error: {e}'

# run_write 是写文件工具，会覆盖目标文件内容。
@tool
def run_write(path: str, content: str) -> str:
    """Write UTF-8 content to a file, replacing its existing content."""
    try:
        file_path = resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f'write {len(content)} bytes to {path}'
    except Exception as e:
        return f'Error: {e}'

# run_edit 是局部替换工具，只替换第一处匹配文本。
@tool
def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace the first exact occurrence of old_text in a UTF-8 file."""
    try:
        file_path = resolve_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f'Error: text not found in {path}'
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f'edit {path}'
    except Exception as e:
        return f'Error: {e}'

# run_glob 用 glob 模式搜索工作区文件。
@tool
def run_glob(pattern: str) -> str:
    """Find workspace files matching a glob pattern."""
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return '\n'.join(results) if results else '(no matches)'
    except Exception as e:
        return f'Error: {e}'

# 兼容不同模型适配器可能返回的内容结构，把 AIMessage 里的文本打印出来。
def print_assistant_message(message: AIMessage) -> None:
    content = message.content
    if isinstance(content, str):
        print(content)
        return
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'text':
                    print(block.get('text', ''))
            elif hasattr(block, 'text'):
                print(block.text)
                
# TOOLS 是交给 Agent 使用的本地工具列表。
TOOLS = [run_bash, run_edit, run_glob, run_write, run_read]
# MIDDLEWARE 定义 Agent 的生命周期插件顺序。
MIDDLEWARE = [
    user_prompt_submit,
# TodoListMiddleware 会注入 write_todos 工具，并把 todo 状态放进 Agent state。
    TodoListMiddleware(
        system_prompt="""
        You must use write_todos for every non-trivial user request.
        Before using any file or shell tool, create a todo list.
        After each step, update todo statuses.
        """), 
    tool_hook, 
    stop_hook
]

MODEL = ChatOpenAI(
    model=MODEL_ID, 
    max_completion_tokens=8000, 
    temperature=0, 
    api_key=OPENAI_API_KEY, 
    base_url=OPENAI_BASE_URL
)
agent = create_agent(
    model=MODEL, 
    tools=TOOLS, 
    system_prompt=SYSTEM, 
    middleware=MIDDLEWARE
)

# 执行一轮 Agent 对话；s05 这里仍然使用 invoke，所以会在本轮结束后一次性拿到结果。
def agent_loop(messages: list) -> None:
    # 把当前完整历史交给 Agent，等 Agent 内部循环结束后返回最终 state。
    result = agent.invoke({'messages': messages})
    new_messages = result['messages'][len(messages):]
    # TodoListMiddleware 会把 todo 状态放在 result["todos"] 里。
    todos = result.get("todos")
    # 如果本轮产生了 todo，就先打印当前 todo 列表。
    if todos:
        print("当前 Todo:")
        for i, todo in enumerate(todos, start=1):
            print(f"{i}. [{todo['status']}] {todo['content']}")
        print()
    for message in new_messages:
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print('模型调用工具:')
            for tool_call in message.tool_calls:
                print('工具名：', tool_call['name'])
                print('参数：', tool_call.get('args', {}))
        elif message.__class__.__name__ == 'ToolMessage':
            print('工具返回结果：')
            print('工具名', getattr(message, 'name', None))
            print('内容:', message.content)
        else:
            print('模型回复:')
            print(getattr(message, 'content', message))
        print()
    messages[:] = result['messages']
    
if __name__ == '__main__':
    print('s05: Todo_write')
    print('输入问题，回车发送。输入 q 退出。\n')
    history = []
    while True:
        try:
            query = input('\x1b[36ms05 >> \x1b[0m')
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ('q', 'exit', ''):
            break
        history.append({'role': 'user', 'content': query})
        agent_loop(history)
