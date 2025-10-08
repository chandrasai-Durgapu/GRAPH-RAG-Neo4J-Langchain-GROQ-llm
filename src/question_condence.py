from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

def format_chat_history(chat_history):
    return [f"Human: {h}\nAI: {a}" for h, a in chat_history]

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that rewrites follow-up questions into standalone questions."),
    ("human", "{chat_history}\nFollow-up question: {question}")
])

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
        RunnablePassthrough.assign(chat_history=lambda x: format_chat_history(x["chat_history"]))
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser()
    ),
    RunnableLambda(lambda x: x["question"]),
)
