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

# Step 3: Configure Train/Test Split
try:
    print("Configuring train/test split for 'journal_classification_pipeline'...")
    result = gds.run_cypher("""
        CALL gds.beta.pipeline.nodeClassification.configureSplit(
            'journal_classification_pipeline',
            {
                testFraction: 0.3,
                validationFolds: 3
            }
        ) YIELD splitConfig
        RETURN splitConfig;
    """)
    print("Train/Test Split Configured:")
    pd.set_option('display.max_colwidth', None)
    print(result)
except Exception as e:
    print(f"Error configuring split: {e}")
    exit(1)

# Step 4: Train the Model
try:
    print("Training model on 'journal_citation_graph'...")
    result = gds.run_cypher("""
        CALL gds.beta.pipeline.nodeClassification.train(
            'journal_citation_graph',
            {
                pipeline: 'journal_classification_pipeline',
                nodeLabels: ['Journal'],
                targetProperty: 'categoryId',
                modelName: 'journal_category_model',
                metrics: ['ACCURACY'],
                randomSeed: 42
            }
        ) YIELD modelInfo
        RETURN modelInfo;
    """)
    print("Training Result:")
    pd.set_option('display.max_colwidth', None)
    print(result)
except Exception as e:
    print(f"Error training model: {e}")
    exit(1)