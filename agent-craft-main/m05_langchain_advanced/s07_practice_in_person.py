from config import OPENAI_API_KEY

from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import tool # 导入 @tool

from langchain_openai import ChatOpenAI

@tool
def get_weather(location):
    """获取对应城市的当前天"""
    return f"{location}当前天气：23℃，晴，风力2级"

#工具表
tools = [get_weather]

def create_agent(llm, prompt):
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
    agent_executor =AgentExecutor(agent=agent, tools=tools)

    store = {}
    def get_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        runnable=agent_executor,
        get_session_history=get_history,
        input_messages_key="input",
        history_messages_key="history"
    )

def run_agent():
    #此处构建llm和prompt
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=OPENAI_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=1.3,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    prompt = ChatPromptTemplate([
        ('system', '你是github的智能助手，帮助用户处理github相关工作，使用简体中文回复'),
        MessagesPlaceholder(variable_name='history'),  # 插入历史对话
        ('user', '{input}'),  # 占位符
        #使用create_tool_calling_agent时，必须加上"agent_scratchpad"，它记录整个ReAct流程（思考-动作-观察），让llm看到，从而做出下步决策
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])

    runnable_with_memory = create_agent(llm,prompt)

    session_id = 'user_123'
    while 1:
        msg = input(">>>")
        if msg == 'q':
            print('bye!')
            break
        print(">>>",end='',flush=True)
        res = runnable_with_memory.invoke(
            {"input": msg},
            config={"configurable": {"session_id": session_id}}
        )


if __name__ == '__main__':
    run_agent()


















