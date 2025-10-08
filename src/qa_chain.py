from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.question_condence import _search_query
from data_retrival.retriever import retriever
from langchain_groq import ChatGroq
from logger.logger import get_logger

logger = get_logger("Answer chain")

try:
    logger.info("Initializing ChatGroq LLM and prompt pipeline.")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.6)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel(
            {
                "context": _search_query | retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("LLM chain pipeline initialized successfully.")

except Exception as e:
    logger.error("Error initializing LLM chain pipeline.", exc_info=True)

def answer_question(question: str, chat_history=None):
    """
    Safely invokes the QA chain with the provided question and optional chat history.

    Args:
        question (str): The user's question.
        chat_history (list): Optional chat history [(user_input, ai_response), ...].

    Returns:
        str: The AI-generated answer, or an error message if something fails.
    """
    try:
        logger.info("Chain invoking begins")
        inputs = {
            "question": question,
            "chat_history": chat_history or []
        }
        logger.info(f"Invoking chain with question: {question}")
        response = chain.invoke(inputs)
        logger.info("Successfully received response from chain.")
        return response
    except Exception as e:
        logger.error(f"Error during chain invocation: {e}", exc_info=True)
        return "⚠️ Sorry, an error occurred while processing your question."
