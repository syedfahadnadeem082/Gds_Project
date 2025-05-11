from graphdatascience import GraphDataScience

# Connect to Neo4j
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

# Drop the graph if it exists (in case of partial state)
try:
    gds.graph.drop("journal_citation_graph", failIfMissing=False)
    print("Dropped existing graph projection 'journal_citation_graph' (if it existed).")
except Exception as e:
    print(f"Error dropping graph: {e}")

# Create the graph projection
try:
    print("Creating graph projection 'journal_citation_graph'...")
    result = gds.run_cypher("""
        CALL gds.graph.project(
            'journal_citation_graph',
            {
                Journal: { label: 'Journal', properties: { categoryId: { defaultValue: -1 } } },
                Paper: { label: 'Paper', properties: { categoryId: { defaultValue: -1 } } }
            },
            {
                CITES: { type: 'CITES', orientation: 'NATURAL' },
                PUBLISHED_IN: { type: 'PUBLISHED_IN', orientation: 'REVERSE' }
            }
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount;
    """)
    print("Graph Projection Result:")
    print(result)
except Exception as e:
    print(f"Error creating graph projection: {e}")
    exit(1)