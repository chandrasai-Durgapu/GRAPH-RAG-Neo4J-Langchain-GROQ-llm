from pyvis.network import Network
from src.neo4j_client import Neo4jClient
from logger.logger import get_logger

logger = get_logger("VisualizationGraph")

def visualize_neo4j_graph(limit=100):
    """
    Visualize a Neo4j graph by fetching nodes and relationships, 
    then displaying it using PyVis.

    Args:
        limit (int): Maximum number of relationships to fetch from Neo4j.

    Returns:
        None
    """
    try:
        logger.info(f"Connecting to Neo4j and fetching up to {limit} relationships")
        neo4j_client = Neo4jClient()
        query = f"""
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT {limit}
        """
        records = neo4j_client.run_query(query)
        logger.info(f"Retrieved {len(records)} relationships from Neo4j")

        net = Network(notebook=True)
        for record in records:
            n = record["n"]
            m = record["m"]
            r = record["r"]
            net.add_node(n.id, label=", ".join(n.labels))
            net.add_node(m.id, label=", ".join(m.labels))
            net.add_edge(n.id, m.id, label=r.type)

        net.show("graph.html")
        logger.info("Graph visualization saved as 'graph.html'")
    except Exception as e:
        logger.error(f"Error during graph visualization: {e}", exc_info=True)
