from graphdatascience import GraphDataScience
import pandas as pd

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

# Drop the existing pipeline if it exists
try:
    gds.run_cypher("""
        CALL gds.beta.pipeline.drop('journal_classification_pipeline', true);
    """)
    print("Dropped existing pipeline 'journal_classification_pipeline' (if it existed).")
except Exception as e:
    print(f"Error dropping pipeline: {e}")

# Drop the existing model if it exists
try:
    gds.run_cypher("""
        CALL gds.beta.model.drop('journal_category_model', true);
    """)
    print("Dropped existing model 'journal_category_model' (if it existed).")
except Exception as e:
    print(f"Error dropping model: {e}")

# Create Node Classification Pipeline
try:
    print("Creating node classification pipeline 'journal_classification_pipeline'...")
    gds.run_cypher("""
        CALL gds.beta.pipeline.nodeClassification.create('journal_classification_pipeline');
    """)
except Exception as e:
    print(f"Error creating pipeline: {e}")
    exit(1)

# Add FastRP Feature Step
try:
    print("Adding FastRP feature step to pipeline...")
    gds.run_cypher("""
        CALL gds.beta.pipeline.nodeClassification.addNodeProperty(
            'journal_classification_pipeline',
            'gds.fastRP.mutate',
            {
                mutateProperty: 'fastRP_embedding',
                embeddingDimension: 256,
                iterationWeights: [0.8, 0.1, 0.1],
                randomSeed: 42
            }
        );
    """)
except Exception as e:
    print(f"Error adding FastRP feature: {e}")
    exit(1)

# Configure Split for Training
try:
    print("Configuring split...")
    gds.run_cypher("""
        CALL gds.beta.pipeline.nodeClassification.configureSplit(
            'journal_classification_pipeline',
            {
                testFraction: 0.3,
                validationFolds: 3
            }
        );
    """)
except Exception as e:
    print(f"Error configuring split: {e}")
    exit(1)

# Add Logistic Regression Model Parameters with Class Weights
try:
    print("Adding logistic regression model parameters...")
    gds.run_cypher("""
        CALL gds.beta.pipeline.nodeClassification.addLogisticRegression(
            'journal_classification_pipeline',
            {
                penalty: 0.01,
                tolerance: 0.001,
                maxEpochs: 100,
                learningRate: 0.001,
                batchSize: 100,
                classWeights: [1.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
            }
        );
    """)
except Exception as e:
    print(f"Error adding logistic regression: {e}")
    exit(1)

# Train the Model on Journal Nodes Only
try:
    print("Training the model 'journal_category_model'...")
    result = gds.run_cypher("""
        CALL gds.beta.pipeline.nodeClassification.train(
            'journal_citation_graph',
            {
                pipeline: 'journal_classification_pipeline',
                targetNodeLabels: ['Journal'],
                modelName: 'journal_category_model',
                targetProperty: 'categoryId',
                randomSeed: 42,
                metrics: ['ACCURACY']
            }
        )
        YIELD modelInfo
        RETURN modelInfo;
    """)
    print("Model Training Result:")
    pd.set_option('display.max_colwidth', None)
    print(result)
except Exception as e:
    print(f"Error training model: {e}")
    exit(1)