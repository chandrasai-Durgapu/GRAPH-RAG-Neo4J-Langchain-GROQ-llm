from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from configuration.config import GROQ_API_KEY
from logger.logger import get_logger

logger = get_logger("Entity Extractor")

class Entities(BaseModel):
    """
    Pydantic model to define the structure of extracted entities.
    """
    names: list[str] = Field(..., description="List of person, organization, or business entities")

try:
    logger.info("Initializing entity extraction chain")

    # Define the prompt for entity extraction
    entity_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are extracting organization and person entities from the text."),
        ("human", "Use the given format to extract information from the following input: {question}")
    ])
    logger.info("Chat prompt template created")

    # LLM setup
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=GROQ_API_KEY
    )
    logger.info("Groq LLM initialized")

    # Chain that produces structured output
    entity_chain = entity_prompt | llm.with_structured_output(Entities)
    logger.info("Entity extraction chain assembled")

except Exception as e:
    logger.error(f"Error during entity_chain setup: {e}", exc_info=True)

def extract_entities(question: str) -> list[str]:
    """
    Extracts named entities (e.g. people, organizations) from a question using a language model.

    Args:
        question (str): The question or input string.

    Returns:
        list[str]: A list of extracted entity names.
    """
    try:
        logger.info(f"Extracting entities from question: {question}")
        entities = entity_chain.invoke({"question": question})
        logger.info(f"Entities found: {entities.names}")
        return entities.names
    except Exception as e:
        logger.error(f"Error extracting entities: {e}", exc_info=True)
        return []
