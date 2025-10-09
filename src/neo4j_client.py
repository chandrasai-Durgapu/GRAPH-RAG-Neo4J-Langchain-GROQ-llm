# neo4j_client.py

from neo4j import GraphDatabase
from configuration.config import NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD
from logger.logger import get_logger
import re

# Initialize logger and global variable declaration
logger = get_logger("Neo4jClient")
neo4j_graph = None # Will be instantiated at the end of the file


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
            # Verify connectivity immediately
            self.driver.verify_connectivity() 
            logger.info("Neo4j driver initialized successfully.")
        except Exception:
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
                # Log only the start of the query for brevity in debug logs
                logger.debug(f"Query ran successfully: {query[:80]}...") 
                return records
        except Exception:
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

    def sanitize_label(self, label):
        """
        Sanitize label or relationship type to be Cypher-safe (no dashes, special chars).
        """
        return re.sub(r'[^A-Za-z0-9_]', '_', label)

    def add_graph_documents(self, graph_documents, baseEntityLabel=True, include_source=True):
        """
        Add graph documents (nodes and relationships) to the Neo4j database.

        Args:
            graph_documents: List of graph documents (nodes + relationships).
            baseEntityLabel (bool): Optionally apply special label treatment.
            include_source (bool): Whether to store source in the DB (future use).
        """
        try:
            with self.driver.session() as session:
                for doc in graph_documents:

                    # Add nodes
                    for node in getattr(doc, 'nodes', []):
                        if hasattr(node, "dict"):
                            node_data = node.dict()
                        elif isinstance(node, dict):
                            node_data = node
                        else:
                            logger.warning(f"Unknown node format: {node}")
                            continue

                        # --- FIX 1: Robust Node ID Lookup ---
                        node_id = node_data.get("id")
                        properties = node_data.get("properties", {})
                        
                        # If ID is not at the top level, check properties
                        if not node_id and properties:
                            node_id = properties.get("id")

                        if not node_id:
                            logger.warning(f"Skipping node with missing ID: {node_data}") 
                            continue
                        
                        labels = node_data.get("labels", [])
                        # --- END FIX 1 ---

                        # Sanitize and join labels
                        safe_labels = ":".join([self.sanitize_label(l) for l in labels])
                        if not safe_labels:
                            safe_labels = "Entity"

                        query = f"""
                        MERGE (n:{safe_labels} {{id: $id}})
                        SET n += $props
                        """
                        session.run(query, {"id": node_id, "props": properties})

                    # Add relationships
                    for rel in getattr(doc, 'relationships', []):
                        if hasattr(rel, "dict"):
                            rel_data = rel.dict()
                        elif isinstance(rel, dict):
                            rel_data = rel
                        else:
                            logger.warning(f"Unknown relationship format: {rel}")
                            continue

                        rel_type = rel_data.get("type", "RELATED_TO")
                        rel_props = rel_data.get("properties", {})
                        source_id = rel_data.get("source_id") 
                        target_id = rel_data.get("target_id")

                        if not source_id or not target_id:
                            logger.warning(f"Skipping relationship with missing endpoints: {rel_data}")
                            continue

                        # Sanitize relationship type
                        safe_rel_type = self.sanitize_label(rel_type)

                        # --- FIX 2: Use MERGE for relationship endpoints ---
                        # Ensures (a) and (b) exist before creating the relationship.
                        query = f"""
                        MERGE (a {{id: $source_id}}) 
                        MERGE (b {{id: $target_id}})
                        MERGE (a)-[r:{safe_rel_type}]->(b)
                        SET r += $props
                        """
                        # --- END FIX 2 ---
                        
                        session.run(query, {
                            "source_id": source_id,
                            "target_id": target_id,
                            "props": rel_props
                        })

                logger.info("Graph documents ingested into Neo4j.")

        except Exception:
            logger.error("Error while adding graph documents to Neo4j.", exc_info=True)


# ====================================================================
# GLOBAL CLIENT INSTANTIATION
# This must be outside the class and at the end of the file.
# ====================================================================

try:
    logger.info("Instantiating global Neo4jClient instance.")
    # This executes the Neo4jClient.__init__ method
    neo4j_graph = Neo4jClient() 
except Exception as e:
    logger.error(f"FATAL: Global Neo4j client failed to instantiate. Graph functionality disabled: {e}", exc_info=True)
# The variable 'neo4j_graph' is now defined (either as an instance or None/failed instance),
# allowing other modules to import it without an ImportError.