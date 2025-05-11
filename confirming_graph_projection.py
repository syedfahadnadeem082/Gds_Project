from graphdatascience import GraphDataScience
import pandas as pd

# Step 1: Connect to Neo4j using the GDS client
try:
    gds = GraphDataScience(
        "bolt://localhost:7687",
        auth=("neo4j", "12345678"),
        database="neo4j"
    )
    print(f"Connected to Neo4j. GDS Version: {gds.version()}")
except Exception as e:
    print(f"Failed to connect to Neo4j: {e}")
    exit(1)

# Step 2: List All Graph Projections
try:
    print("Listing all graph projections...")
    result = gds.run_cypher("""
        CALL gds.graph.list()
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount;
    """)
    print("Graph Projections:")
    print(result)
except Exception as e:
    print(f"Error listing graphs: {e}")
    exit(1)