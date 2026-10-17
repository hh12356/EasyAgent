#!/usr/bin/env python3
"""
s12_cron_scheduler.py - Cron Scheduler

    +--------------------------+   09:00   +-----------------------+
    | 0 9 * * *               | --------> | [Scheduled] run tests |
    | prompt: "run tests"      |           +-----------+-----------+
    +--------------------------+                       |
          scheduled_jobs                    cron_queue | agent idle
                                                        v
                                                +-------------+
                                                | Agent Loop  |
                                                +-------------+
"""

import glob
import json
import os
import re
import secrets
import subprocess
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
DURABLE_PATH = WORKDIR / ".scheduled_tasks.json" # 持久化文件位置
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

ENVIRONMENT_PROMPT = (
    "Windows: the bash tool runs through cmd.exe; use cmd.exe syntax, not Unix "
    "Bash or PowerShell syntax, and prefer dedicated file tools for file operations"
    if os.name == "nt"
    else "Unix-like: the bash tool runs the system shell"
)
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
    "Use tools to solve tasks. "
    "Use schedule_cron for work that should start at a future local time."
)


# -- From s04: tool implementations --

def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True, errors="replace",
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            return f"Error: command exited with status {result.returncode}\n{output}"
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        lines = file_path.read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as error:
        return f"Error: {error}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as error:
        return f"Error: {error}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = (WORKDIR / path).resolve()
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as error:
        return f"Error: {error}"


def run_glob(pattern: str) -> str:
    try:
        matches = sorted({
            match
            for match in glob.glob(pattern, root_dir=WORKDIR, recursive=True)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
        })
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) if shown else "(no matches)"
    except Exception as error:
        return f"Error: {error}"


TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string"}},
                      "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "old_text": {"type": "string"},
                                     "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern; ** matches recursively.",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"]}},
]

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}


# -- From s04: hooks and permission checks --

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}


def register_hook(event: str, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def contains_destructive_command(command: str) -> bool:
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))


def request_permission(block, reason: str) -> str | None:
    if threading.current_thread() is not threading.main_thread():
        return "Permission denied: scheduled turns cannot request interactive approval"

    print(f"\n\033[33m[permission] {reason}\033[0m")
    print(f"   Tool: {block.name}({block.input})")
    choice = input("   Allow? [y/N] ").strip().lower()
    if choice not in ("y", "yes"):
        return "Permission denied by user"
    return None


def permission_hook(block):
    if block.name == "bash":
        command = block.input.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                print(f"\n\033[31m[blocked] '{pattern}'\033[0m")
                return "Permission denied by deny list"
        if contains_destructive_command(command) or any(
            keyword in command for keyword in DESTRUCTIVE
        ):
            return request_permission(block, "Potentially destructive command")

    if block.name in ("read_file", "write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            return request_permission(block, "Access outside workspace")
    return None


def log_hook(block):
    preview = str(list(block.input.values())[:2])[:60]
    print(f"\033[90m[HOOK] {block.name}({preview})\033[0m")
    return None


def large_output_hook(block, output):
    if len(str(output)) > 100000:
        print(
            f"\033[33m[HOOK] Large output from {block.name}: "
            f"{len(str(output))} chars\033[0m"
        )
    return None


def context_inject_hook(query: str):
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None


def summary_hook(messages: list):
    tool_count = sum(
        1
        for message in messages
        for block in (
            message.get("content")
            if isinstance(message.get("content"), list)
            else []
        )
        if isinstance(block, dict) and block.get("type") == "tool_result"
    )
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None


register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


# -- New in s12: cron jobs --

@dataclass
class CronJob:
    id: str
    cron: str # 定义触发任务时间
    prompt: str
    recurring: bool # 是否重复执行
    durable: bool # 持久化开关
    pending_delivery: bool = False # 待投递标签
    last_fired: str | None = None # 上次触发时间


scheduled_jobs: dict[str, CronJob] = {}
cron_queue: list[CronJob] = []
cron_lock = threading.RLock()


# 用来判断cron段中的某一段跟当前数值是否匹配
def _cron_field_matches(field: str, value: int) -> bool:
    # * 代表任意值都匹配
    if field == "*":
        return True
    # 间隔语法
    if field.startswith("*/"):
        return value % int(field[2:]) == 0
    # 多值列表则递归调用自己
    if "," in field:
        # 任意匹配就返回True
        return any(_cron_field_matches(part.strip(), value)
                   for part in field.split(","))
    # 范围区间
    if "-" in field:
        start, end = field.split("-", 1)
        return int(start) <= value <= int(end)
    # 保底直接判断
    return value == int(field)


def cron_matches(cron_expr: str, moment: datetime) -> bool:
    fields = cron_expr.strip().split()
    # cron长度必须为5
    if len(fields) != 5:
        return False

    # 序列解包
    minute, hour, day, month, weekday = fields
    # datetime.weekday()从0开始计
    cron_weekday = (moment.weekday() + 1) % 7
    # 判断除天和星期之外的所有项
    if not (
        _cron_field_matches(minute, moment.minute)
        and _cron_field_matches(hour, moment.hour)
        and _cron_field_matches(month, moment.month)
    ):
        return False

    # 单独判断(这两个是OR关系)
    day_matches = _cron_field_matches(day, moment.day)
    weekday_matches = _cron_field_matches(weekday, cron_weekday)
    if day == "*" and weekday == "*":
        return True
    if day == "*":
        return weekday_matches
    if weekday == "*":
        return day_matches
    return day_matches or weekday_matches


# 递归校验单条 cron 字段格式与数值范围(无错误返回None，报错返回str)
def _validate_cron_field(field: str, minimum: int, maximum: int) -> str | None:
    if field == "*":
        return None
    if field.startswith("*/"):
        step = field[2:]
        # 步长不是纯数字或步长小于0报错
        if not step.isdigit() or int(step) <= 0:
            return f"Invalid step: {field}"
        return None
    # 分段递归验证
    if "," in field:
        for part in field.split(","):
            error = _validate_cron_field(part.strip(), minimum, maximum)
            # 但凡有一个报错直接返回
            if error:
                return error
        # 全部通过
        return None
    # 范围类型
    if "-" in field:
        start, end = field.split("-", 1)
        if not start.isdigit() or not end.isdigit():
            return f"Invalid range: {field}"
        start_value, end_value = int(start), int(end)
        if start_value > end_value:
            return f"Range start is greater than end: {field}"
        if start_value < minimum or end_value > maximum:
            return f"Range {field} is outside [{minimum}-{maximum}]"
        return None
    # 除上述情况，cron必须为数字
    if not field.isdigit():
        return f"Invalid field: {field}"
    value = int(field)
    if value < minimum or value > maximum:
        return f"Value {value} is outside [{minimum}-{maximum}]"
    return None


# 验证完整cron
def validate_cron(cron_expr: str) -> str | None:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"

    field_rules = [
        ("minute", 0, 59),
        ("hour", 0, 23),
        ("day-of-month", 1, 31),
        ("month", 1, 12),
        ("day-of-week", 0, 6),
    ]
    for field, (name, minimum, maximum) in zip(fields, field_rules):
        error = _validate_cron_field(field, minimum, maximum)
        if error:
            return f"{name}: {error}"
    return None


def save_durable_jobs():
    with cron_lock:
        payload = [
            # asdict()把dataclass实例对象，转换成普通Python字典
            asdict(job)
            for job in scheduled_jobs.values()
            if job.durable # 取出需要持久化的job
        ]
        # 创建临时文件，成功写入再替换原文件
        temporary = DURABLE_PATH.with_name(
            f"{DURABLE_PATH.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, DURABLE_PATH)
        finally:
            temporary.unlink(missing_ok=True) # 报错则删除临时文件，改写失败


# 加载持久化job
def load_durable_jobs():
    if not DURABLE_PATH.exists():
        return
    try:
        payload = json.loads(DURABLE_PATH.read_text(encoding="utf-8"))
        # 必须为jsonList
        if not isinstance(payload, list):
            raise ValueError("expected a JSON list")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"  [cron] could not load {DURABLE_PATH.name}: {error}")
        return

    loaded = 0
    with cron_lock:
        for item in payload:
            try:
                job = CronJob(**item)
                error = validate_cron(job.cron)
                if error:
                    raise ValueError(error)
                if not job.id.startswith("cron_"):
                    raise ValueError("invalid job ID")
                if not job.prompt.strip():
                    raise ValueError("prompt cannot be empty")
            except (TypeError, ValueError) as error:
                print(f"  [cron] skipped invalid saved job: {error}")
                continue
            # 加入执行目录
            scheduled_jobs[job.id] = job
            if job.pending_delivery:
                cron_queue.append(job)
            loaded += 1
    if loaded:
        print(f"  [cron] loaded {loaded} durable job(s)")


# 新建cron_id
def new_cron_id() -> str:
    for _ in range(100):
        job_id = f"cron_{secrets.token_hex(4)}"
        if job_id not in scheduled_jobs:
            return job_id
    raise RuntimeError("Could not allocate a cron job ID")


# 新增job
def schedule_job(cron: str, prompt: str, recurring: bool = True,
                 durable: bool = True) -> CronJob | str:
    error = validate_cron(cron)
    if error:
        return error
    if not prompt.strip():
        return "Prompt cannot be empty"

    with cron_lock:
        job = CronJob(
            id=new_cron_id(),
            cron=cron,
            prompt=prompt,
            recurring=recurring,
            durable=durable,
        )
        scheduled_jobs[job.id] = job
        try:
            if durable:
                save_durable_jobs()
        except Exception:
            scheduled_jobs.pop(job.id, None)
            raise
    print(f"  [cron] scheduled {job.id}: {cron} -> {prompt[:60]}")
    return job


# 删除job
def cancel_job(job_id: str) -> str:
    with cron_lock:
        job = scheduled_jobs.get(job_id)
        if job is None:
            return f"Job {job_id} not found"

        previous_queue = list(cron_queue) # 备份cron_queue
        scheduled_jobs.pop(job_id)
        cron_queue[:] = [queued for queued in cron_queue if queued.id != job_id]
        try:
            # 如果是该删除任务被持久化需更新持久化文件
            if job.durable:
                save_durable_jobs()
        # 报错还原
        except Exception:
            scheduled_jobs[job_id] = job
            cron_queue[:] = previous_queue
            raise
    print(f"  [cron] cancelled {job_id}")
    return f"Cancelled {job_id}"


# 将到期任务入队
def _enqueue_due_job(job: CronJob, minute_marker: str | None = None): # minute_marker用来记录本次任务触发的时间标记
    old_pending = job.pending_delivery
    old_last_fired = job.last_fired
    # 更新投递状态
    job.pending_delivery = True
    if minute_marker is not None:
        job.last_fired = minute_marker
    try:
        # 若持久化则更新持久化文件
        if job.durable:
            save_durable_jobs()
    # 报错还原
    except Exception:
        job.pending_delivery = old_pending
        job.last_fired = old_last_fired
        raise
    # 加入队列
    cron_queue.append(job)


# 轮询任务加入待处理
def poll_due_jobs(moment: datetime):
    # 生成规范化时间字符串(丢掉秒)
    minute_marker = moment.strftime("%Y-%m-%d %H:%M")
    with cron_lock:
        for job in list(scheduled_jobs.values()):
            try:
                # 任务已处理
                if job.pending_delivery or job.last_fired == minute_marker:
                    continue
                # 判断并加入可处理任务
                if cron_matches(job.cron, moment):
                    _enqueue_due_job(job, minute_marker)
                    print(f"  [cron] due {job.id}: {job.prompt[:60]}")
            except Exception as error:
                print(f"  [cron] could not enqueue {job.id}: {error}")


# 取出队列中的待处理任务
def consume_cron_queue() -> list[CronJob]:
    with cron_lock:
        jobs = list(cron_queue)
        cron_queue.clear()
    return jobs


# 完成任务收尾:循环任务重置 pending 标记，一次性任务直接删除
def acknowledge_cron_jobs(jobs: list[CronJob]):
    changed: list[tuple[CronJob, bool]] = []
    removed: list[CronJob] = []
    with cron_lock:
        for delivered in jobs:
            current = scheduled_jobs.get(delivered.id)
            if current is None:
                continue
            changed.append((current, current.pending_delivery))
            if current.recurring:
                current.pending_delivery = False
            else:
                removed.append(current)
                scheduled_jobs.pop(current.id)

        try:
            if any(job.durable for job, _ in changed):
                save_durable_jobs()
        except Exception:
            for job in removed:
                scheduled_jobs[job.id] = job
            for job, pending in changed:
                job.pending_delivery = pending
            queued_ids = {job.id for job in cron_queue}
            for job, _ in changed:
                if job.id not in queued_ids:
                    cron_queue.append(job)
            raise


# 恢复一批 Cron 定时任务，把标记为已投递的任务重新放回待执行队列
def restore_cron_jobs(jobs: list[CronJob]):
    with cron_lock:
        queued_ids = {job.id for job in cron_queue}
        for delivered in jobs:
            current = scheduled_jobs.get(delivered.id)
            if current is None:
                continue
            current.pending_delivery = True
            if current.id not in queued_ids:
                cron_queue.append(current)
                queued_ids.add(current.id)


# 判断待执行队列是否为空
def has_cron_queue() -> bool:
    with cron_lock:
        return bool(cron_queue)


# 创建cron定时任务
def run_schedule_cron(cron: str, prompt: str, recurring: bool = True,
                      durable: bool = True) -> str:
    result = schedule_job(cron, prompt, recurring, durable)
    # schedule_job()报错时返回字符串
    if isinstance(result, str):
        return f"Error: {result}"
    return f"Scheduled {result.id}: {cron} -> {prompt}"


# 读取全部定时任务
def run_list_crons() -> str:
    with cron_lock:
        jobs = list(scheduled_jobs.values())
    if not jobs:
        return "No cron jobs."

    lines = []
    for job in jobs:
        frequency = "recurring" if job.recurring else "one-shot"
        storage = "durable" if job.durable else "session"
        lines.append(
            f"{job.id}: {job.cron} -> {job.prompt[:60]} "
            f"[{frequency}, {storage}]"
        )
    return "\n".join(lines)


# 删除任务
def run_cancel_cron(job_id: str) -> str:
    return cancel_job(job_id)


# 加入cron相关工具
TOOLS.extend([
    {"name": "schedule_cron",
     "description": "Schedule a prompt with a 5-field cron expression.",
     "input_schema": {"type": "object",
                      "properties": {
                          "cron": {"type": "string"},
                          "prompt": {"type": "string"},
                          "recurring": {"type": "boolean"},
                          "durable": {"type": "boolean"}},
                      "required": ["cron", "prompt"]}},
    {"name": "list_crons", "description": "List scheduled cron jobs.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "cancel_cron", "description": "Cancel a cron job by ID.",
     "input_schema": {"type": "object",
                      "properties": {"job_id": {"type": "string"}},
                      "required": ["job_id"]}},
])

# 字典没有append(),追加使用update()
TOOL_HANDLERS.update({
    "schedule_cron": run_schedule_cron,
    "list_crons": run_list_crons,
    "cancel_cron": run_cancel_cron,
})


def execute_tool(block) -> str:
    blocked = trigger_hooks("PreToolUse", block)
    if blocked is not None:
        return str(blocked)

    handler = TOOL_HANDLERS.get(block.name)
    try:
        output = handler(**block.input) if handler else f"Unknown: {block.name}"
    except Exception as error:
        output = f"Error: {error}"
    trigger_hooks("PostToolUse", block, output)
    return str(output)


# -- Scheduler and agent loop --

RUNTIME_STOP = threading.Event() # 用来通知所有后台运行线程停止
runtime_threads: list[threading.Thread] = [] # 保存所有启动的后台运行线程对象
runtime_started = False # 记录运行时是否已经启动
runtime_lock = threading.Lock()
agent_lock = threading.Lock()
session_history: list = []


# 每秒轮询一次，检查有没有到期的定时任务需要触发；外部调用RUNTIME_STOP.set()，循环在下一次 wait 时退出
def cron_scheduler_loop(stop_event: threading.Event = RUNTIME_STOP):
    while not stop_event.wait(1.0):
        poll_due_jobs(datetime.now())


def agent_loop(messages: list, context: dict | None = None):
    fired = consume_cron_queue() # 待处理任务
    scheduled_start = len(messages)
    # 将待执行任务加入上下文
    for job in fired:
        messages.append({"role": "user", "content": f"[Scheduled] {job.prompt}"})
        print(f"  [cron] delivered {job.id}: {job.prompt[:60]}")

    waiting_for_ack = list(fired)
    while True:
        try:
            response = client.messages.create(
                model=MODEL,
                system=SYSTEM,
                messages=messages,
                tools=TOOLS,
                max_tokens=8000,
            )
        # 报错时把上下文中的任务导回列表
        except Exception as error:
            if waiting_for_ack:
                del messages[scheduled_start:]
                restore_cron_jobs(waiting_for_ack)
            print(f"  [error] {type(error).__name__}: {error}")
            return context

        messages.append({"role": "assistant", "content": response.content})
        if waiting_for_ack:
            try:
                # cron完成收尾
                acknowledge_cron_jobs(waiting_for_ack)
            except Exception as error:
                print(f"  [cron] acknowledgement failed: {error}")
            waiting_for_ack = []

        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return context

        results = []
        for block in tool_calls:
            output = execute_tool(block)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        messages.append({"role": "user", "content": results})


# 打印llm回复内容
def print_latest_assistant_text(messages: list):
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            print(content)
        else:
            for block in content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    print(block.get("text", ""))
        return

# 执行agent_loop
def run_agent_turn_locked(user_query: str | None = None):
    if user_query is not None:
        trigger_hooks("UserPromptSubmit", user_query) # Hook
        session_history.append({"role": "user", "content": user_query})
    agent_loop(session_history)
    print_latest_assistant_text(session_history)
    print()


# 后台执行任务
def queue_processor_loop(stop_event: threading.Event = RUNTIME_STOP):
    while not stop_event.wait(0.2):
        if not has_cron_queue() or not agent_lock.acquire(blocking=False): # 查看队列是否有内容和锁是否被占用
            continue
        try:
            # cron到时时自动执行
            if has_cron_queue():
                run_agent_turn_locked()
        finally:
            agent_lock.release()


# 启动轮询执行进程
def start_runtime_threads():
    global runtime_started
    with runtime_lock:
        # 已被启动则退出
        if runtime_started:
            return
        load_durable_jobs()
        RUNTIME_STOP.clear()
        runtime_threads.extend([
            # 后台持续轮询
            threading.Thread(
                target=cron_scheduler_loop,
                name="cron-scheduler",
                daemon=True,
            ),
            threading.Thread(
                target=queue_processor_loop,
                name="cron-queue-processor",
                daemon=True,
            ),
        ])
        for thread in runtime_threads:
            thread.start()
        runtime_started = True


# 停止后台进程
def stop_runtime_threads():
    global runtime_started
    with runtime_lock:
        if not runtime_started:
            return
        RUNTIME_STOP.set() # 退出后台执行
        for thread in runtime_threads:
            thread.join(timeout=1) # 等待该进程结束，最多等一秒
        runtime_threads.clear()
        runtime_started = False


if __name__ == "__main__":
    print("s12: Cron Scheduler - run prompts on a local schedule")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    start_runtime_threads()
    try:
        while True:
            try:
                # \001/\002 tell Readline the ANSI escapes have zero display width.
                query = input("\001\033[36m\002s12 >> \001\033[0m\002")
            except (EOFError, KeyboardInterrupt):
                break
            if query.strip().lower() in ("q", "exit", ""):
                break
            with agent_lock:
                run_agent_turn_locked(query)
    finally:
        stop_runtime_threads()
