from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from configuration.config import GROQ_API_KEY
from src.neo4j_client import Neo4jClient
from logger.logger import get_logger

logger = get_logger("GraphIngestor")


def ingest_documents_to_neo4j(documents):
    logger.info("Starting document ingestion to Neo4j...")

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=GROQ_API_KEY
        )
        transformer = LLMGraphTransformer(llm=llm)

        logger.debug("Converting documents to graph documents...")
        graph_docs = transformer.convert_to_graph_documents(documents)
        logger.info(f"Converted {len(graph_docs)} graph documents")

    except Exception as e:
        logger.exception("Failed during LLM document transformation")
        return

    neo4j_client = Neo4jClient()

    try:
        logger.debug("Creating Neo4j constraints and indexes...")

        neo4j_client.run_query("CREATE CONSTRAINT IF NOT EXISTS ON (d:Document) ASSERT d.id IS UNIQUE")
        neo4j_client.run_query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (n) ON EACH [n.name, n.text]")

        logger.debug("Ingesting graph documents into Neo4j...")
        neo4j_client.add_graph_documents(
            graph_documents=graph_docs,
            baseEntityLabel=True,
            include_source=True
        )
        logger.info("Graph documents successfully ingested into Neo4j")

    except Exception as e:
        logger.exception("Failed while inserting data into Neo4j")

    finally:
        neo4j_client.close()
        logger.info("Neo4j connection closed")

