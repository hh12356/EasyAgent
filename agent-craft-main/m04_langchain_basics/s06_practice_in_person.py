from config import OPENAI_API_KEY
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI

def create_agent(llm, prompt):
    parser = StrOutputParser()
    chain = prompt | llm | parser

    store = {}
    def get_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        runnable=chain,
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
        temperature=1.3
    )
    prompt = ChatPromptTemplate([
        ('system', '你是github的智能助手，帮助用户处理github相关工作，使用简体中文回复'),
        MessagesPlaceholder(variable_name='history'),  # 插入历史对话
        ('user', '{input}')  # 占位符
    ])

    runnable_with_memory = create_agent(llm,prompt)

    session_id = 'user_123'
    while 1:
        msg = input(">>>")
        if msg == 'q':
            print('bye!')
            break
        res = runnable_with_memory.invoke(
            {"input": msg},
            config={"configurable": {"session_id": session_id}}
        )
        print(f">>>{res}")


if __name__ == '__main__':
    run_agent()


















