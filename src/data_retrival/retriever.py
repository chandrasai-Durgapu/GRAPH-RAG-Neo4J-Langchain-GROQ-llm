import re
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from configuration.config import NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD
from entity_extractor import extract_entities
from logger.logger import get_logger
from neo4j_client import Neo4jClient


# Initialize Logger
logger = get_logger("Retriever")

# --- INITIALIZATION ---
# Initialize embeddings and vector index
# This section is wrapped in a try/except block to catch connection errors (as seen in the original trace)
try:
    logger.info(" Initializing HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    logger.info(" Connecting to Neo4jVector index...")
    # This is the line that failed in the original trace, ensure your NEO4J_URL, USERNAME, and PASSWORD are correct.
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
    # Define a placeholder for the Neo4j Graph Client, assuming it's used elsewhere
    # Note: The original ImportError was due to 'neo4j_graph' not being defined/imported here.
    # The structure suggests the main function passes a 'Neo4jClient' instance, but 
    # if 'neo4j_graph' is intended to be exported from this file, it should be initialized here.
    # We will proceed by only exporting the 'retriever' function and assuming the client is passed to it.
    
except Exception as e:
    logger.exception(" Failed to initialize Neo4jVector.")
    # Define vector_index as None or a mock object to allow the file to be imported, 
    # but the retriever function will fail when using it.
    vector_index = None


# --- UTILITY FUNCTIONS ---

def remove_lucene_chars(text: str) -> str:
    """
    Removes special characters from the query string that are not compatible with Lucene full-text search.
    """
    return re.sub(r'[-+!(){}[\]^"~*?:\\/]|&&|\|\|', ' ', text)


def generate_full_text_query(input: str) -> str:
    """
    Converts a natural language input into a fuzzy full-text search query string.
    """
    try:
        if not input or not isinstance(input, str):
            logger.warning(" Invalid input received for Lucene query generation.")
            return ""

        cleaned = remove_lucene_chars(input)
        words = [word.strip() for word in cleaned.split() if word.strip()]
        # Fuzzy search with ~2 tolerance
        query = " AND ".join(f"{word}~2" for word in words) if words else ""

        logger.debug(f" Generated Lucene query: {query}")
        return query

    except Exception as e:
        logger.exception(f" Failed to generate Lucene full-text query for input: {input}")
        return ""


# --- RETRIEVAL LOGIC ---

def structured_retriever(graph: Neo4jClient, question: str, entities: List[str]) -> str:
    """
    Performs structured graph-based retrieval from Neo4j using entity mentions.
    
    Uses an improved Cypher query to get facts as clear triples for the LLM.
    """
    result = []

    for entity in entities:
        query = generate_full_text_query(entity)
        if not query:
             continue

        # --- IMPROVED CYPHER QUERY ---
        # 1. Searches for the top 1 entity node ('entity' index).
        # 2. Matches relationships in both directions (one hop).
        # 3. Uses COALESCE to prioritize 'name' or 'id' for better context.
        # 4. Limits facts to 10 per entity to prevent context overload.
        cypher = """
        CALL db.index.fulltext.queryNodes('entity', $query, {limit: 1}) 
        YIELD node, score
        MATCH (node)-[r]-(neighbor)
        WHERE type(r) <> 'MENTIONS'
        WITH 
            node, 
            neighbor,
            r,
            score,
            coalesce(node.name, node.id, 'UNKNOWN_ENTITY') AS entity_name

        // Construct the output as a clear triple representation
        RETURN 
            '(' + entity_name + ':' + labels(node)[0] + ') ' + 
            '-[' + type(r) + ']-' + 
            ' (' + coalesce(neighbor.name, neighbor.id, 'UNKNOWN_NEIGHBOR') + ':' + labels(neighbor)[0] + ')' AS output
        ORDER BY score DESC
        LIMIT 10
        """

        try:
            logger.debug(f" Querying structured data for entity: {entity} with Cypher query: {query}")
            records = graph.run_query(cypher, {"query": query})
            
            if records:
                result.append(f"Entity: {entity}")
                result.extend(r["output"] for r in records)
                result.append("-" * 20)
                
        except Exception as e:
            logger.exception(f" Error querying entity '{entity}'")
            result.append(f"\n Error querying entity '{entity}': {e}\n")

    return "\n".join(result).strip()


def retriever(graph: Neo4jClient, question: str) -> str:
    """
    Combines both structured (Neo4j graph) and unstructured (vector search) retrieval
    to provide a comprehensive answer to the user's question.
    """
    logger.info(f" Received question: {question}")

    # 1. Entity extraction
    try:
        entities = extract_entities(question)
        logger.info(f" Extracted entities: {entities}")
    except Exception as e:
        logger.exception(" Failed to extract entities")
        return " Failed to extract entities."

    # 2. Structured graph search
    try:
        structured = structured_retriever(graph, question, entities)
    except Exception as e:
        logger.exception(" Structured retrieval failed")
        structured = " Structured retrieval failed."
        
    # 3. Unstructured vector similarity search
    unstructured = " Unstructured retrieval failed: Vector index not initialized."
    if vector_index:
        try:
            # Check if vector_index was initialized successfully
            unstructured_docs = vector_index.similarity_search(question, k=5)
            # Use 'text' property from Document node for content
            unstructured = "\n".join(doc.page_content for doc in unstructured_docs)
        except Exception as e:
            logger.exception(" Vector similarity search failed")
            unstructured = " Unstructured retrieval failed."
    
    # 4. Combine and return
    return f""" Structured Data (Facts):
{structured}

---
Unstructured Data (Document Chunks):
{unstructured}"""