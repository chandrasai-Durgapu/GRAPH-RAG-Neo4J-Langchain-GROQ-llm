import re
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from configuration.config import NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD
from src.entity_extractor import extract_entities
from logger.logger import get_logger
from src.neo4j_client import Neo4jClient

logger = get_logger("Retriever")

# Initialize embeddings and vector index
try:
    logger.info(" Initializing HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    logger.info(" Connecting to Neo4jVector index...")
    vector_index = Neo4jVector.from_existing_graph(
        embedding=embeddings,
        url=NEO4J_URL,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    logger.info(" Neo4jVector initialized.")
except Exception as e:
    logger.exception(" Failed to initialize Neo4jVector.")


def remove_lucene_chars(text: str) -> str:
    """
    Removes special characters from the query string that are not compatible with Lucene full-text search.

    Args:
        text (str): Raw query string.

    Returns:
        str: Cleaned query string.
    """
    return re.sub(r'[-+!(){}[\]^"~*?:\\/]|&&|\|\|', ' ', text)


def generate_full_text_query(input: str) -> str:
    """
    Converts a natural language input into a fuzzy full-text search query string.

    Args:
        input (str): Input string or entity.

    Returns:
        str: Formatted Lucene query string for fuzzy search.
    """
    try:
        if not input or not isinstance(input, str):
            logger.warning(" Invalid input received for Lucene query generation.")
            return ""

        cleaned = remove_lucene_chars(input)
        words = [word.strip() for word in cleaned.split() if word.strip()]
        query = " AND ".join(f"{word}~2" for word in words) if words else ""

        logger.debug(f" Generated Lucene query: {query}")
        return query

    except Exception as e:
        logger.exception(f" Failed to generate Lucene full-text query for input: {input}")
        return ""



def structured_retriever(graph: Neo4jClient, question: str, entities: List[str]) -> str:
    """
    Performs structured graph-based retrieval from Neo4j using entity mentions.

    Args:
        graph (Neo4jClient): An instance of the Neo4j client.
        question (str): User's natural language question.
        entities (List[str]): List of named entities to use in full-text search.

    Returns:
        str: Structured knowledge graph facts related to the question.
    """
    result = ""

    for entity in entities:
        query = generate_full_text_query(entity)

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
            logger.debug(f" Querying structured data for entity: {entity}")
            records = graph.run_query(cypher, {"query": query})
            result += f"\n Entity: {entity}\n"
            result += "\n".join(r["output"] for r in records) + "\n"
        except Exception as e:
            logger.exception(f" Error querying entity '{entity}'")
            result += f"\n Error querying entity '{entity}': {e}\n"

    return result.strip()


def retriever(graph: Neo4jClient, question: str) -> str:
    """
    Combines both structured (Neo4j graph) and unstructured (vector search) retrieval
    to provide a comprehensive answer to the user's question.

    Args:
        graph (Neo4jClient): Neo4j database client.
        question (str): User input question.

    Returns:
        str: Combined output from structured and unstructured data sources.
    """
    logger.info(f" Received question: {question}")

    # Entity extraction
    try:
        entities = extract_entities(question)
        logger.info(f" Extracted entities: {entities}")
    except Exception as e:
        logger.exception(" Failed to extract entities")
        return " Failed to extract entities."

    # Structured graph search
    try:
        structured = structured_retriever(graph, question, entities)
    except Exception as e:
        logger.exception(" Structured retrieval failed")
        structured = " Structured retrieval failed."

    # Unstructured vector similarity search
    try:
        unstructured_docs = vector_index.similarity_search(question, k=5)
        unstructured = "\n".join(doc.page_content for doc in unstructured_docs)
    except Exception as e:
        logger.exception(" Vector similarity search failed")
        unstructured = " Unstructured retrieval failed."

    return f""" Structured data:
{structured}

Unstructured data:
{unstructured}"""
