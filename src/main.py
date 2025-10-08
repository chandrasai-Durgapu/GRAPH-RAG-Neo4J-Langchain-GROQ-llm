from src.wikipedia_loader import load_wikipedia_docs
from data_ingestion.graph_ingestion import ingest_documents_to_neo4j
from src.neo4j_client import Neo4jClient
from src.qa_chain import answer_question
from logger.logger import get_logger

logger = get_logger("Main")

def main():
    """
    Entry point for loading documents, ingesting into Neo4j,
    and running example QA queries.
    """
    neo4j_client = None

    try:
        logger.info("Starting Wikipedia document load...")
        docs = load_wikipedia_docs("Elizabeth I")
        logger.info(f"Loaded {len(docs)} document chunks.")

        logger.info("Ingesting documents into Neo4j...")
        ingest_documents_to_neo4j(docs)
        logger.info("Data ingestion complete.")

        logger.info("Running example questions...")

        question1 = "Which house did Elizabeth I belong to?"
        answer1 = answer_question(question1)
        logger.info(f"Q: {question1}\nA: {answer1}\n")

        question2 = "When was she born?"
        answer2 = answer_question(question2, chat_history=[(question1, answer1)])
        logger.info(f"Q: {question2}\nA: {answer2}")

    except Exception as e:
        logger.error("An error occurred during execution.", exc_info=True)

    finally:
        if neo4j_client:
            neo4j_client.close()
            logger.info("Neo4j connection closed.")

if __name__ == "__main__":
    main()
