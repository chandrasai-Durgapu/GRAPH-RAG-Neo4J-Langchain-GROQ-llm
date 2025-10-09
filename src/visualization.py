from pyvis.network import Network
from neo4j_client import Neo4jClient
from logger.logger import get_logger

logger = get_logger("VisualizationGraph")

def visualize_neo4j_graph(limit=100):
    try:
        logger.info(f"Connecting to Neo4j and fetching up to {limit} relationships")
        neo4j_client = Neo4jClient()
        
        # --- FIX: Structured Cypher Query to ensure consistent dictionary output ---
        query = f"""
        MATCH (n)-[r]->(m)
        RETURN 
            {{id: n.id, label: head(labels(n)), name: n.name}} AS n, 
            {{type: type(r)}} AS r, 
            {{id: m.id, label: head(labels(m)), name: m.name}} AS m
        LIMIT {limit}
        """
        
        records = neo4j_client.run_query(query) 
        logger.info(f"Retrieved {len(records)} relationships from Neo4j")

        net = Network(notebook=True, directed=True)

        for record in records:
            # These are now guaranteed to be dictionaries due to the query structure
            n = record.get("n", {})
            m = record.get("m", {})
            r = record.get("r", {})
            
            # --- FIX: Simplified Python Logic using dictionary access ---
            
            # Use the custom 'id' property for pyvis nodes
            n_id = n.get("id")
            m_id = m.get("id")

            # Use 'name' or the primary label for visualization text
            n_label = n.get("name") or n.get("label", "Entity")
            m_label = m.get("name") or m.get("label", "Entity")
            
            # Relationship type
            rel_type = r.get("type", "RELATED_TO")
            
            # Skip if critical IDs are missing (e.g., node was not properly ingested)
            if not n_id or not m_id:
                logger.warning(f"Skipping record due to missing node ID: {record}")
                continue

            # Ensure IDs are strings for pyvis
            str_n_id = str(n_id)
            str_m_id = str(m_id)

            # Add nodes with labels and titles for hover text
            net.add_node(str_n_id, label=n_label, title=n_label)
            net.add_node(str_m_id, label=m_label, title=m_label)
            
            # Add edge
            net.add_edge(str_n_id, str_m_id, label=rel_type, title=rel_type)

        net.show("graph.html")
        logger.info("Graph visualization saved as 'graph.html'")
        
    except Exception as e:
        logger.error(f"Error during graph visualization: {e}", exc_info=True)