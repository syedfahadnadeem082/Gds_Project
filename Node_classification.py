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

# Step 2: Drop the Pipeline if it Exists
try:
    print("Dropping pipeline 'journal_classification_pipeline' if it exists...")
    result = gds.run_cypher("""
        CALL gds.beta.pipeline.drop('journal_classification_pipeline')
        YIELD pipelineName
        RETURN pipelineName;
    """)
    print("Pipeline dropped (if it existed).")
    print("Drop Result:")
    print(result)
except Exception as e:
    print(f"Pipeline may not exist or error dropping pipeline: {e}")

# Step 3: Create a New Pipeline
try:
    print("Creating new node classification pipeline...")
    result = gds.run_cypher("""
        CALL gds.beta.pipeline.nodeClassification.create('journal_classification_pipeline')
        YIELD name
        RETURN name;
    """)
    print("Pipeline 'journal_classification_pipeline' created successfully.")
    print("Creation Result:")
    print(result)
except Exception as e:
    print(f"Error creating pipeline: {e}")
    exit(1)

# Step 4: Configure the Pipeline with Computed FastRP Embeddings
try:
    print("Configuring pipeline with computed FastRP embeddings...")
    result = gds.run_cypher("""
        CALL gds.beta.pipeline.nodeClassification.addNodeProperty(
            'journal_classification_pipeline',
            'fastRP',
            {
                embeddingDimension: 128,
                iterationWeights: [0.0, 1.0, 1.0],
                randomSeed: 42,
                mutateProperty: 'fastRP_embedding'
            }
        ) YIELD nodePropertySteps
        RETURN nodePropertySteps;
    """)
    print("Computed FastRP embeddings added as node property to pipeline.")
    print("Configuration Result:")
    pd.set_option('display.max_colwidth', None)
    print(result)
except Exception as e:
    print(f"Error configuring pipeline: {e}")
    exit(1)

# Step 5: Verify Pipeline Configuration (Fixed)
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