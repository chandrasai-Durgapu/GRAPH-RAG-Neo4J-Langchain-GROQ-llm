import os
import re
from typing import List, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
from langchain_groq import ChatGroq

from logger.logger import get_logger  # Your custom logger module
from src.visualization import visualize_neo4j_graph

# === Load environment variables ===
load_dotenv()

NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logger = get_logger("MainApp")

# Validate critical environment variables
if not all([NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD, GROQ_API_KEY]):
    logger.error("One or more environment variables (NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD, GROQ_API_KEY) are missing.")
    raise EnvironmentError("Missing required environment variables. Please check your .env file.")

# === Connect to Neo4j Graph ===
try:
    graph = Neo4jGraph(
        url=NEO4J_URL,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )
    logger.info("Connected to Neo4j successfully.")
except Exception as e:
    logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
    raise


# === Load Wikipedia Data ===
def load_wikipedia_docs(query: str, max_pages: int = 3):
    """
    Load Wikipedia documents based on a query.

    Args:
        query (str): The Wikipedia page to load.
        max_pages (int): Max number of documents to return.

    Returns:
        List of documents.
    """
    try:
        loader = WikipediaLoader(query=query)
        raw_docs = loader.load()
        logger.info(f"Wikipedia docs loaded for query '{query}': {len(raw_docs)} docs")
        return raw_docs[:max_pages]
    except Exception as e:
        logger.error(f"Failed to load Wikipedia docs for query '{query}': {e}", exc_info=True)
        return []


# === Split documents into chunks ===
def split_documents(documents, chunk_size=512, chunk_overlap=25):
    """
    Split large documents into smaller chunks.

    Args:
        documents (List): List of document objects.
        chunk_size (int): Number of tokens per chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List of document chunks.
    """
    try:
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = splitter.split_documents(documents)
        logger.info(f"Documents split into {len(split_docs)} chunks.")
        return split_docs
    except Exception as e:
        logger.error(f"Failed to split documents: {e}", exc_info=True)
        return []


# === Initialize LLM ===
try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=GROQ_API_KEY,
    )
    logger.info("Initialized ChatGroq LLM.")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
    raise


# === Graph ingestion function ===
def ingest_graph(documents):
    """
    Convert documents into graph format and ingest into Neo4j.

    Args:
        documents (List): List of document chunks.
    """
    try:
        from langchain_experimental.graph_transformers import LLMGraphTransformer

        transformer = LLMGraphTransformer(llm=llm)
        graph_documents = transformer.convert_to_graph_documents(documents)

        graph.add_graph_documents(
            graph_documents=graph_documents,
            baseEntityLabel=True,
            include_source=True,
        )
        logger.info(f"Ingested {len(graph_documents)} graph documents into Neo4j.")
    except Exception as e:
        logger.error(f"Graph ingestion failed: {e}", exc_info=True)


# === Setup Neo4j Vector Store with embeddings ===
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_index = Neo4jVector.from_existing_graph(
        embedding=embeddings,
        url=NEO4J_URL,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
    )
    logger.info("Neo4jVector index initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Neo4jVector: {e}", exc_info=True)
    vector_index = None


# === Entity extraction model and prompt ===
class Entities(BaseModel):
    names: List[str] = Field(..., description="Person, organization, or business entities in text")


entity_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are extracting organization and person entities from the text."),
        ("human", "Extract entities from the following input: {question}"),
    ]
)

try:
    entity_chain = entity_extraction_prompt | llm.with_structured_output(Entities)
    logger.info("Entity extraction chain initialized.")
except Exception as e:
    logger.error(f"Failed to initialize entity extraction chain: {e}", exc_info=True)
    entity_chain = None


# === Helper functions for query cleaning ===
def remove_lucene_chars(text: str) -> str:
    """
    Remove special Lucene characters from input to prevent query errors.

    Args:
        text (str): Input text.

    Returns:
        Cleaned text.
    """
    try:
        cleaned = re.sub(r'[-+!(){}[\]^"~*?:\\/]|&&|\|\|', ' ', text)
        return cleaned
    except Exception as e:
        logger.error(f"Error cleaning lucene chars: {e}", exc_info=True)
        return text


def generate_full_text_query(input_str: str) -> str:
    """
    Converts a natural language input into a fuzzy Lucene full-text search query.

    Args:
        input_str (str): Input string or entity.

    Returns:
        str: Formatted Lucene query string.
    """
    try:
        cleaned = remove_lucene_chars(input_str)
        words = [w.strip() for w in cleaned.split() if w.strip()]
        if not words:
            return ""
        return " AND ".join(f"{word}~2" for word in words)
    except Exception as e:
        logger.error(f"Error generating full text query: {e}", exc_info=True)
        return ""


# === Structured retriever querying Neo4j fulltext ===
def structured_retriever(question: str) -> str:
    """
    Query Neo4j structured graph data for entities related to the question.

    Args:
        question (str): User question.

    Returns:
        str: Concatenated string of structured results.
    """
    if entity_chain is None:
        logger.error("Entity extraction chain is not initialized.")
        return " Entity extraction unavailable."

    try:
        result = ""
        entities = entity_chain.invoke({"question": question})
        logger.info(f"Extracted entities: {entities.names}")

        for entity in entities.names:
            fulltext_query = generate_full_text_query(entity)

            cypher = """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit: 2})
            YIELD node, score
            
            CALL {
                WITH node
                MATCH (node)-[r]->(neighbor)
                WHERE type(r) <> 'MENTIONS'
                RETURN node.id + ' -[' + type(r) + ']-> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r]-(neighbor)
                WHERE type(r) <> 'MENTIONS'
                RETURN neighbor.id + ' <-[' + type(r) + ']- ' + node.id AS output
            }
            
            RETURN output
            LIMIT 50
            """

            try:
                response = graph.query(cypher, {"query": fulltext_query})
                result += f"\n Entity: {entity}\n"
                result += "\n".join(el["output"] for el in response) + "\n"
            except Exception as e:
                logger.error(f"Error querying entity '{entity}': {e}", exc_info=True)
                result += f"\n Error querying entity '{entity}': {e}\n"

        return result.strip()
    except Exception as e:
        logger.error(f"Structured retrieval failed: {e}", exc_info=True)
        return " Structured retrieval failed."


# === Combined retriever for structured + vector search ===
def retriever(question: str) -> str:
    """
    Retrieve relevant information from structured Neo4j graph and unstructured vector index.

    Args:
        question (str): User question.

    Returns:
        str: Combined structured and unstructured data.
    """
    logger.info(f"Received question for retrieval: {question}")

    structured_data = structured_retriever(question)

    unstructured_data = ""
    try:
        if vector_index:
            unstructured_docs = vector_index.similarity_search(question, k=5)
            unstructured_data = "\n".join(doc.page_content for doc in unstructured_docs)
        else:
            unstructured_data = " Vector index not initialized."
    except Exception as e:
        logger.error(f"Vector similarity search failed: {e}", exc_info=True)
        unstructured_data = " Unstructured retrieval failed."

    combined = f"""Structured data:
{structured_data}

Unstructured data:
{unstructured_data}"""
    return combined


# === Chat history formatter for follow-up question condensing ===
def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List[str]:
    """
    Format chat history tuples into strings for input to prompt.

    Args:
        chat_history (List[Tuple[str, str]]): List of (human, ai) turns.

    Returns:
        List[str]: Formatted chat history strings.
    """
    buffer = []
    for human, ai in chat_history:
        buffer.append(f"Human: {human}")
        buffer.append(f"AI: {ai}")
    return buffer


# === Prompt to condense follow-up questions ===
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that rewrites follow-up questions into standalone questions.",
        ),
        (
            "human",
            "{chat_history}\nFollow-up question: {question}",
        ),
    ]
)


# === RunnableBranch to handle chat history condensing ===
_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
        RunnablePassthrough.assign(chat_history=lambda x: _format_chat_history(x["chat_history"]))
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    RunnableLambda(lambda x: x["question"]),
)


# === Final QA prompt template ===
QA_PROMPT_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""

prompt = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)


# === Final QA chain combining retriever and LLM ===
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


# === Main execution example ===
if __name__ == "__main__":
    try:
        # Load and ingest Wikipedia docs (run once)
        logger.info("Loading Wikipedia documents for 'Elizabeth I'...")
        docs = load_wikipedia_docs("Elizabeth I")
        if not docs:
            logger.warning("No Wikipedia documents loaded, exiting.")
            exit(1)

        split_docs = split_documents(docs)
        if not split_docs:
            logger.warning("Document splitting failed or returned no chunks, exiting.")
            exit(1)

        logger.info("Ingesting graph data to Neo4j...")
        ingest_graph(split_docs)
        try:
            logger.info("Visualizing Neo4j graph...")
            visualize_neo4j_graph(limit=100)
            logger.info("Visualization completed.")
        except Exception as e:
            logger.error(f"Failed to visualize graph: {e}", exc_info=True)
            logger.info("Running example questions...")

        # Simple query without chat history
        question1 = "Which house did Elizabeth I belong to?"
        logger.info(f"Asking question: {question1}")
        answer1 = chain.invoke({"question": question1})
        logger.info(f"Answer: {answer1}")
        print(f"\nQuestion: {question1}\nAnswer: {answer1}")

        # Follow-up question with chat history
        question2 = "When was she born?"
        chat_history = [(question1, answer1)]
        logger.info(f"Asking follow-up question: {question2}")
        answer2 = chain.invoke({"question": question2, "chat_history": chat_history})
        logger.info(f"Answer: {answer2}")
        print(f"\nFollow-up Question: {question2}\nAnswer: {answer2}")

    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        print(f"An error occurred: {e}")
