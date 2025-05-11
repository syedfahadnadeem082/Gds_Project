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

# Step 2: Verify Pipeline Configuration
try:
    print("Verifying configuration of 'journal_classification_pipeline'...")
    result = gds.run_cypher("""
        CALL gds.beta.pipeline.list('journal_classification_pipeline')
        YIELD pipelineInfo
        RETURN pipelineInfo;
    """)
    print("Pipeline Info:")
    pd.set_option('display.max_colwidth', None)
    print(result)
    
    # Extract and verify nodePropertySteps from featurePipeline
    if not result.empty and 'pipelineInfo' in result.columns:
        pipeline_info = result['pipelineInfo'][0]
        node_property_steps = pipeline_info.get('featurePipeline', {}).get('nodePropertySteps', None)
        print("\nNode Property Steps:")
        print(node_property_steps)
        if node_property_steps is None or len(node_property_steps) == 0:
            raise Exception("Pipeline configuration missing nodePropertySteps!")
except Exception as e:
    print(f"Error verifying pipeline: {e}")
    exit(1)

# Step 3: Add a Logistic Regression Model to the Pipeline
try:
    print("Adding logistic regression model to 'journal_classification_pipeline'...")
    gds.run_cypher("""
        CALL gds.beta.pipeline.nodeClassification.addLogisticRegression(
            'journal_classification_pipeline',
            {
                penalty: 0.01,
                maxEpochs: 100
            }
        )
    """)
    print("Logistic regression model added to pipeline.")
except Exception as e:
    print(f"Error adding model: {e}")
    exit(1)

# Step 4: Verify Model Addition
try:
    print("Verifying model addition in 'journal_classification_pipeline'...")
    result = gds.run_cypher("""
        CALL gds.beta.pipeline.list('journal_classification_pipeline')
        YIELD pipelineInfo
        RETURN pipelineInfo.trainingParameterSpace AS trainingParameterSpace;
    """)
    print("Training Parameter Space:")
    pd.set_option('display.max_colwidth', None)
    print(result)
    
    # Extract and verify LogisticRegression parameters
    if not result.empty and 'trainingParameterSpace' in result.columns:
        training_params = result['trainingParameterSpace'][0]
        logistic_params = training_params.get('LogisticRegression', [])
        print("\nLogistic Regression Parameters:")
        print(logistic_params)
        if not logistic_params:
            raise Exception("Logistic Regression model not added to pipeline!")
except Exception as e:
    print(f"Error verifying model addition: {e}")
    exit(1)