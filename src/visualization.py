from pyvis.network import Network
from src.neo4j_client import Neo4jClient

def visualize_neo4j_graph(limit=100):
    neo4j_client = Neo4jClient()
    query = f"""
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    LIMIT {limit}
    """
    records = neo4j_client.run_query(query)

    net = Network(notebook=True)
    for record in records:
        n = record["n"]
        m = record["m"]
        r = record["r"]
        net.add_node(n.id, label=str(n.labels))
        net.add_node(m.id, label=str(m.labels))
        net.add_edge(n.id, m.id, label=r.type)

    net.show("graph.html")
