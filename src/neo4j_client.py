from neo4j import GraphDatabase
from configuration.config import NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD
from logger.logger import get_logger

logger = get_logger("Neo4jClient")

class Neo4jClient:
    """
    A client for interacting with a Neo4j database.
    """

    def __init__(self):
        """
        Initialize Neo4j driver using credentials from config.
        """
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URL,
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
            )
            logger.info("Neo4j driver initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize Neo4j driver.", exc_info=True)
            raise

    def close(self):
        """
        Close the Neo4j driver.
        """
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed.")

    def run_query(self, query, parameters=None):
        """
        Run a Cypher query with optional parameters.

        Args:
            query (str): Cypher query to execute.
            parameters (dict): Optional dictionary of parameters.

        Returns:
            List[dict]: Query results as a list of dictionaries.
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                records = [record.data() for record in result]
                logger.debug(f"Query ran successfully: {query}")
                return records
        except Exception as e:
            logger.error(f"Error running query: {query}", exc_info=True)
            return []

    def get_schema(self):
        """
        Retrieve node labels and their counts in the Neo4j graph.

        Returns:
            List[dict]: Node labels with their occurrence counts.
        """
        query = """
        MATCH (n)
        RETURN labels(n) AS labels, count(*) AS count
        ORDER BY count DESC
        LIMIT 10
        """
        return self.run_query(query)
