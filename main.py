import os
from openai import OpenAI
import json
import subprocess

TOOLS = [
  {
    "type": "function",
    "function": {
      "name": "run_bash",
      "description": "在服务器上执行一段bash shell命令，并返回命令输出结果",
      "parameters": {
        "type": "object",
        "properties": {
          "command": {
            "type": "string",
            "description": "要执行的bash命令文本"
          }
        },
        "required": ["command"],
      }
    }
  }
]

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"], 
    base_url="https://api.deepseek.com"
    )

#初始对话
messages = [
    {"role": "system", "content": "你是一个能执行命令的助手。注意：这台机器是 Windows，shell 是 cmd，不要用 ls/grep 这类 Unix 命令。"},
    {"role": "user", "content": "看看当前目录有哪些文件,哪个文件最大"}
]

for i in range(10):

    resp = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        tools=TOOLS
    )
    msg = resp.choices[0].message
    print(f"---- resp{i} -----")
    print(msg.content)
    print(msg.tool_calls)
    print("----usage----")
    print(resp.usage.prompt_tokens)
    print("\n")

    #无条件append
    messages.append(msg)

    #是否为工具调用轮
    if msg.tool_calls:
        for call in msg.tool_calls:

            #取出工具调用
            call_id = call.id
            func_args = json.loads(call.function.arguments)
            cmd = func_args["command"]

            #执行bash,subprocess,超时+文本+截断
            try:
                proc = subprocess.run(
                    cmd,
                    #交给系统shell(bash)
                    shell=True,
                    #输出到.stdout和.stderr
                    capture_output=True,
                    #解码成str
                    text=True,
                    timeout=10
                )
                tool_out = (proc.stdout or '') + "\nstderr:\n" + (proc.stderr or '')
            except subprocess.TimeoutExpired:
                tool_out="【超时】命令执行超过10秒已终止"

            #截断：最多3000字
            max_len=3000
            if len(tool_out)>max_len:
                tool_out = tool_out[:max_len]+"\n...【输出截断】..."

            #append tool_out
            messages.append({
                "role":"tool",
                "tool_call_id":call_id,
                "content":tool_out
            })

    else:
        question = input()
        if question == "exit":
            print("----end----")
            exit()
        else:
            messages.append({"role": "user", "content": question})

print("到轮数上限了")