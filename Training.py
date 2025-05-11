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

# Step 2: Drop Graph Projection if it Exists
try:
    print("Dropping graph projection 'journal_citation_graph' if it exists...")
    gds.run_cypher("""
        CALL gds.graph.drop('journal_citation_graph', false)
        YIELD graphName
        RETURN graphName;
    """)
    print("Graph projection dropped (if it existed).")
except Exception as e:
    print(f"Graph may not exist or error dropping graph: {e}")

# Step 3: Recreate Graph Projection with Correct Default Value
try:
    print("Recreating graph projection 'journal_citation_graph' with default categoryId = -1...")
    result = gds.run_cypher("""
        CALL gds.graph.project(
            'journal_citation_graph',
            ['Journal', 'Paper'],
            {
                PUBLISHED_IN: {orientation: 'REVERSE'},
                CITES: {orientation: 'NATURAL'}
            },
            {
                nodeProperties: {
                    categoryId: {defaultValue: -1, propertyType: 'Integer'}
                }
            }
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount;
    """)
    print("Graph projection 'journal_citation_graph' created successfully.")
    print("Projection Result:")
    print(result)
except Exception as e:
    print(f"Error creating graph projection: {e}")
    exit(1)

# Step 4: Inspect Graph Projection Schema
try:
    print("\nInspecting graph projection schema for 'journal_citation_graph'...")
    result = gds.run_cypher("""
        CALL gds.graph.list('journal_citation_graph')
        YIELD graphName, nodeCount, relationshipCount, schema
        RETURN graphName, nodeCount, relationshipCount, schema;
    """)
    print("Graph Projection Details:")
    pd.set_option('display.max_colwidth', None)
    print(result)
    
    # Extract node labels and properties from schema
    if not result.empty and 'schema' in result.columns:
        schema = result['schema'][0]
        node_labels = list(schema.get('nodes', {}).keys())
        node_properties = schema.get('nodes', {})
        print("\nNode Labels in Graph:")
        print(node_labels)
        print("\nNode Properties by Label:")
        print(node_properties)
except Exception as e:
    print(f"Error inspecting graph projection: {e}")
    exit(1)