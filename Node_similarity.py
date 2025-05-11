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

# Step 1: Drop existing SIMILAR_TO and SIMILAR_TO_NEW relationships if they exist
try:
    print("Dropping existing SIMILAR_TO and SIMILAR_TO_NEW relationships...")
    gds.run_cypher("""
        MATCH ()-[r:SIMILAR_TO]->()
        DELETE r;
    """)
    gds.run_cypher("""
        MATCH ()-[r:SIMILAR_TO_NEW]->()
        DELETE r;
    """)
    print("Dropped existing SIMILAR_TO and SIMILAR_TO_NEW relationships (if any).")
except Exception as e:
    print(f"Error dropping relationships: {e}")

# Step 2: Drop the existing graph projection if it exists
try:
    print("Dropping existing graph projection 'journal_paper_field_graph'...")
    gds.graph.drop("journal_paper_field_graph", failIfMissing=False)
    print("Dropped existing graph projection 'journal_paper_field_graph' (if it existed).")
except Exception as e:
    print(f"Error dropping graph: {e}")

# Step 3: Create a graph projection with PUBLISHED_IN relationships
try:
    print("Creating graph projection 'journal_paper_field_graph'...")
    G, result = gds.graph.project(
        "journal_paper_field_graph",
        {
            "Paper": {"label": "Paper"},
            "Journal": {"label": "Journal", "properties": ["categoryId"]}
        },
        {
            "PUBLISHED_IN": {"type": "PUBLISHED_IN", "orientation": "NATURAL"}
        },
        readConcurrency=4
    )
    print("Graph Projection Result:")
    print(result)
except Exception as e:
    print(f"Error creating graph projection: {e}")
    exit(1)

# Step 4: Deduplicate Journal nodes with identical names (normalized)
try:
    print("Deduplicating Journal nodes with identical names (after normalization)...")
    result = gds.run_cypher("""
        MATCH (j:Journal)
        WITH TRIM(LOWER(REPLACE(j.Journal_Name, '&', 'and'))) AS normalizedName, collect(j) AS journals
        WHERE size(journals) > 1
        WITH normalizedName, journals
        // Prioritize a journal with categoryId != -1, otherwise take the first one
        WITH normalizedName, journals,
             head([j IN journals WHERE j.categoryId <> -1 | j]) AS keepJournal,
             [j IN journals WHERE j.categoryId <> -1 | j] AS validJournals,
             [j IN journals WHERE j <> head([j2 IN journals WHERE j2.categoryId <> -1 | j2]) | j] AS toDelete
        WHERE keepJournal IS NOT NULL
        // If no journal with categoryId != -1, take the first journal
        WITH normalizedName, journals, 
             coalesce(keepJournal, head(journals)) AS finalKeepJournal,
             toDelete
        // Transfer relationships to the kept journal
        UNWIND toDelete AS deleteJournal
        MATCH (deleteJournal)<-[r:PUBLISHED_IN]-(p:Paper)
        WHERE NOT (p)-[:PUBLISHED_IN]->(finalKeepJournal)
        CREATE (p)-[:PUBLISHED_IN]->(finalKeepJournal)
        WITH normalizedName, journals, deleteJournal
        DETACH DELETE deleteJournal
        WITH normalizedName, size(journals) AS originalCount
        RETURN normalizedName AS name, originalCount, 1 AS keptCount;
    """)
    print("Deduplication Result (Journal Name, Original Count, Kept Count):")
    print(result)
except Exception as e:
    print(f"Error deduplicating journals: {e}")
    exit(1)

# Step 5: Compute SIMILAR_TO relationships based on shared papers using Cypher
try:
    print("Computing SIMILAR_TO relationships based on shared papers...")
    result = gds.run_cypher("""
        MATCH (j1:Journal)<-[:PUBLISHED_IN]-(p1:Paper)
        MATCH (j2:Journal)<-[:PUBLISHED_IN]-(p2:Paper)
        WHERE id(j1) < id(j2) AND j1.Journal_Name <> j2.Journal_Name
        WITH j1, j2, 
             collect(DISTINCT p1) AS papers1, 
             collect(DISTINCT p2) AS papers2
        WITH j1, j2, 
             size([p IN papers1 WHERE p IN papers2]) AS intersection,
             size(papers1) + size(papers2) - size([p IN papers1 WHERE p IN papers2]) AS union
        WHERE union > 0
        WITH j1, j2, toFloat(intersection) / union AS jaccard
        WHERE jaccard > 0.0
        CREATE (j1)-[r:SIMILAR_TO {weight: jaccard}]->(j2)
        RETURN j1.Journal_Name AS j1Name, j2.Journal_Name AS j2Name, r.weight AS jaccardScore
        LIMIT 10;
    """)
    print("SIMILAR_TO Relationships Based on Jaccard Similarity:")
    print(result)
except Exception as e:
    print(f"Error computing SIMILAR_TO relationships: {e}")
    exit(1)

# Step 6: Create a new graph projection including SIMILAR_TO relationships
try:
    print("Creating new graph projection with SIMILAR_TO relationships...")
    gds.graph.drop("journal_paper_field_graph", failIfMissing=False)  # Drop the old projection
    G, result = gds.graph.project(
        "journal_paper_field_graph",
        {
            "Journal": {"label": "Journal", "properties": ["categoryId"]}
        },
        {
            "SIMILAR_TO": {"type": "SIMILAR_TO", "orientation": "NATURAL", "properties": ["weight"]}
        },
        readConcurrency=4
    )
    print("New Graph Projection Result:")
    print(result)
except Exception as e:
    print(f"Error creating new graph projection: {e}")
    exit(1)

# Step 7: Split the Journal nodes into 85% train and 15% test
try:
    print("Splitting Journal nodes into 85% train and 15% test...")
    result = gds.run_cypher("""
        MATCH (j:Journal)
        WITH j, rand() AS r
        ORDER BY r
        WITH collect(j) AS journals
        WITH journals, size(journals) AS total, toInteger(0.15 * size(journals)) AS testSize
        UNWIND range(0, total-1) AS idx
        WITH journals[idx] AS journal, idx < (total - testSize) AS isTrain
        SET journal.isTest = NOT isTrain
        RETURN count(*) AS splitCount;
    """)
    print("Data Split Result:")
    print(result)
except Exception as e:
    print(f"Error splitting data: {e}")
    exit(1)

# Step 8: Reset categoryId for test nodes to simulate prediction
try:
    print("Resetting categoryId for test nodes...")
    result = gds.run_cypher("""
        MATCH (j:Journal {isTest: true})
        SET j.originalCategoryId = j.categoryId
        SET j.categoryId = -1
        RETURN count(*) AS resetCount;
    """)
    print("CategoryId Reset Result:")
    print(result)
except Exception as e:
    print(f"Error resetting categoryId: {e}")
    exit(1)

# Step 9: Compute Node Similarity on precomputed SIMILAR_TO relationships
try:
    print("Computing node similarity on precomputed SIMILAR_TO relationships...")
    result = gds.run_cypher("""
        CALL gds.nodeSimilarity.write('journal_paper_field_graph', {
            nodeLabels: ['Journal'],
            relationshipTypes: ['SIMILAR_TO'],
            relationshipWeightProperty: 'weight',
            writeRelationshipType: 'SIMILAR_TO_NEW',
            writeProperty: 'similarityScore',
            similarityCutoff: 0.0
        })
        YIELD computeMillis, nodesCompared, relationshipsWritten
        RETURN computeMillis, nodesCompared, relationshipsWritten;
    """)
    print("Node Similarity Result:")
    print(result)
except Exception as e:
    print(f"Error computing node similarity: {e}")
    exit(1)

# Step 10: Check the number of SIMILAR_TO_NEW relationships
try:
    print("Checking SIMILAR_TO_NEW relationships between test and train journals...")
    result = gds.run_cypher("""
        MATCH (j1:Journal {isTest: true})-[r:SIMILAR_TO_NEW]->(j2:Journal {isTest: false})
        RETURN count(r) AS relationshipCount;
    """)
    print("Number of SIMILAR_TO_NEW Relationships (Test to Train):")
    print(result)
except Exception as e:
    print(f"Error checking SIMILAR_TO_NEW relationships: {e}")
    exit(1)

# Step 11: Predict categoryId for Test Journals based on similarity
try:
    print("Predicting categoryId for test journals based on similarity...")
    result = gds.run_cypher("""
        MATCH (j1:Journal {isTest: true})
        MATCH (j1)-[r:SIMILAR_TO_NEW]->(j2:Journal {isTest: false})
        WHERE j2.categoryId IS NOT NULL AND j2.categoryId <> -1
        WITH j1, j2, r.similarityScore AS score
        ORDER BY score DESC
        LIMIT 1
        SET j1.categoryId = j2.categoryId
        RETURN j1.Journal_Name AS Journal, j1.originalCategoryId AS OriginalCategoryId, j1.categoryId AS PredictedCategoryId;
    """)
    print("Similarity-based Predictions for Test Journals:")
    pd.set_option('display.max_colwidth', None)
    print(result)
except Exception as e:
    print(f"Error predicting categoryId: {e}")
    exit(1)