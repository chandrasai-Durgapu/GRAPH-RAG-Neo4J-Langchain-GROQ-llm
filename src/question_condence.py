from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from logger.logger import get_logger


from logger.logger import get_logger

logger = get_logger("ChatHistoryFormatter")

def format_chat_history(chat_history):
    """
    Format chat history into a structured list of strings for prompt input.

    Args:
        chat_history (list): A list of (human, ai) message tuples.

    Returns:
        list[str]: A list where each element is a formatted string like:
                   "Human: <question>\nAI: <answer>"
    """
    try:
        if not chat_history:
            logger.info("No chat history provided.")
            return []
        
        formatted = [f"Human: {h}\nAI: {a}" for h, a in chat_history]
        logger.info(f"Formatted {len(formatted)} entries of chat history.")
        return formatted
    except Exception as e:
        logger.error(f"Failed to format chat history: {e}", exc_info=True)
        return []



try:
    CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that rewrites follow-up questions into standalone questions."),
        ("human", "{chat_history}\nFollow-up question: {question}")
    ])

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)

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
    logger.info("Runnable search query pipeline initialized successfully.")
except Exception as e:
    logger.error("Failed to initialize the Runnable search query pipeline.", exc_info=True)
    logger.error(e)