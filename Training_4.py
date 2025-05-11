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

# Step 2: Verify Trained Model
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

# Step 3: Predict with the Model (Using Mutate)
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

# Step 4: Retrieve Predicted Categories
try:
    print("Retrieving predicted categories for Journal nodes...")
    result = gds.run_cypher("""
        MATCH (j:Journal)
        WHERE j.predictedCategory IS NOT NULL
        RETURN j.Journal_Name AS journalName, j.predictedCategory AS predictedCategory, j.categoryId AS actualCategoryId
        LIMIT 10;
    """)
    print("Prediction Results (Top 10):")
    pd.set_option('display.max_colwidth', None)
    print(result)
except Exception as e:
    print(f"Error retrieving predictions: {e}")
    exit(1)