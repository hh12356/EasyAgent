from config import OPENAI_API_KEY
import os
from embeddings import get_embeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_tool_calling_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.callbacks import StreamingStdOutCallbackHandler

def build_rag_chain(llm):

    persist_directory = './chroma_db_war_and_peace_bge_small_en_v1.5'
    embedding_model_name = 'BAAI/bge-small-en-v1.5'
    encoder_model_name = "BAAI/bge-reranker-base"

    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f'索引目录{persist_directory}未找到，请先运行 build_index.py')

    #链接向量数据库
    embedding_model = get_embeddings(model_name=embedding_model_name, device="cpu")
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    # R 检索
    #粗召回
    base_retriever = db.as_retriever(search_kwargs={'k':50})
    #精排序
    encoder = HuggingFaceCrossEncoder(model_name=encoder_model_name)
    reranker = CrossEncoderReranker(model=encoder,top_n=6)
    #管道封装
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=reranker
    )
    retriever = compression_retriever

    # A 增强
    sys_prompt ="""
    你是一个博学的历史学家和文学评论家。
    请根据以下上下文回答问题。如果上下文**暗示**了答案，即使未明说，也可推理回答。
    如果完全无关，请回答“对不起，根据所提供的上下文我不知道”。

    [上下文]: {context}
    [问题]: {question}
    """
    prompt = ChatPromptTemplate([
        ("system",sys_prompt),
        ("user","{question}")
    ])

    def format_docs(docs):
        return '\n'.join(doc.page_content for doc in docs)

    # RAG
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        |prompt
        |llm
        |StrOutputParser()
    )

    return rag_chain

def create_agent():
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=OPENAI_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=1.3,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    prompt = ChatPromptTemplate([
        ('system', '你是一个博学的文学评论家，使用简体中文回复。你的所有知识只能来自工具返回，不允许瞎编，如实说不知道'),
        MessagesPlaceholder(variable_name='history'),  # 插入历史对话
        ('user', '{input}'),  # 占位符
        # 使用create_tool_calling_agent时，必须加上"agent_scratchpad"，它记录整个ReAct流程（思考-动作-观察），让llm看到，从而做出下步决策
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ])

    @tool
    def get_weather(location):
        """获取对应城市的当前天"""
        return f"{location}当前天气：23℃，晴，风力2级"

    rag_chain = build_rag_chain(llm)

    @tool
    def get_war_and_peace(query):
        """查询《战争与和平》小说中的内容，包括人物、情节、历史事件等。query参数必须为英文，否则无法返回有效结果"""
        print("RAG")
        return rag_chain.invoke(query)

    # 工具表
    tools = [get_weather,get_war_and_peace]

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

    runnable_with_memory = create_agent()

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
        print()


if __name__ == '__main__':
    run_agent()


















