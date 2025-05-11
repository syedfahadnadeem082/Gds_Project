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

# Step 2: Drop and Recreate the Graph Projection
try:
    print("Dropping existing graph projection 'journal_citation_graph'...")
    gds.graph.drop("journal_citation_graph", failIfMissing=False)
    print("Dropped existing graph projection 'journal_citation_graph' (if it existed).")
except Exception as e:
    print(f"Error dropping graph: {e}")

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

# Step 3: Verify Trained Model
try:
    print("Verifying trained model 'journal_category_model'...")
    result = gds.run_cypher("""
        CALL gds.beta.model.list()
        YIELD modelInfo
        WHERE modelInfo.modelName = 'journal_category_model'
        RETURN modelInfo;
    """)
    print("Model Info:")
    pd.set_option('display.max_colwidth', None)
    print(result)
except Exception as e:
    print(f"Error verifying model: {e}")
    exit(1)

# Step 4: Predict with the Model (Using Mutate)
try:
    print("Predicting categories on 'journal_citation_graph' with mutate...")
    result = gds.run_cypher("""
        CALL gds.beta.pipeline.nodeClassification.predict.mutate(
            'journal_citation_graph',
            {
                pipeline: 'journal_classification_pipeline',
                nodeLabels: ['Journal'],
                modelName: 'journal_category_model',
                mutateProperty: 'predictedCategory'
            }
        )
        YIELD nodePropertiesWritten, mutateMillis, postProcessingMillis, preProcessingMillis, computeMillis
        RETURN nodePropertiesWritten, mutateMillis, postProcessingMillis, preProcessingMillis, computeMillis;
    """)
    print("Mutation Result:")
    pd.set_option('display.max_colwidth', None)
    print(result)
except Exception as e:
    print(f"Error predicting categories: {e}")
    exit(1)

# Step 5: Write Predicted Categories to Database
try:
    print("Writing predicted categories to the database...")
    result = gds.run_cypher("""
        CALL gds.graph.writeNodeProperties(
            'journal_citation_graph',
            ['predictedCategory'],
            ['Journal']
        )
        YIELD propertiesWritten
        RETURN propertiesWritten;
    """)
    print("Write Result:")
    pd.set_option('display.max_colwidth', None)
    print(result)
except Exception as e:
    print(f"Error writing predicted categories: {e}")
    exit(1)

# Step 6: Retrieve Predicted Categories
try:
    print("Retrieving predicted categories for Journal nodes...")
    result = gds.run_cypher("""
        MATCH (j:Journal)
        WHERE j.predictedCategory IS NOT NULL
        RETURN j.Journal_Name AS journalName, j.predictedCategory AS predictedCategory, j.categoryId AS actualCategoryId;
    """)
    print("Prediction Results (Top 10):")
    pd.set_option('display.max_colwidth', None)
    print(result)
except Exception as e:
    print(f"Error retrieving predictions: {e}")
    exit(1)