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

# Step 2: Add a Logistic Regression Model to the Pipeline
try:
    print("Adding logistic regression model to 'journal_classification_pipeline'...")
    result = gds.run_cypher("""
        CALL gds.beta.pipeline.nodeClassification.addLogisticRegression(
            'journal_classification_pipeline',
            {
                penalty: 0.01,
                maxEpochs: 100
            }
        ) YIELD modelInfo
        RETURN modelInfo;
    """)
    print("Logistic regression model added to pipeline.")
    print("Model Configuration Result:")
    pd.set_option('display.max_colwidth', None)
    print(result)
except Exception as e:
    print(f"Error adding model: {e}")
    exit(1)