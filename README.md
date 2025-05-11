Journal Classification Using Graph Data Science
This repository contains the implementation of a graph-based machine learning project to classify journals into subject categories using a bibliographic dataset. The project leverages Neo4j Graph Data Science (GDS) version 2.13.4 to apply two approaches: Node Similarity with Jaccard similarity and FastRP embeddings combined with logistic regression. The dataset, derived from migration research [1], includes 126 journals, 30,441 papers, and 111,283 relationships (PUBLISHED_IN and CITES). The FastRP approach achieved a test accuracy of 31.6%, but uniform predictions (category 0: Social Sciences) highlighted challenges due to a small dataset and class imbalance. This README provides a comprehensive guide to understand the project and reproduce the results.
Project Goals

Classify journals into subject categories (e.g., Social Sciences, Natural Sciences, Life Sciences & Medicine, Engineering & Technology, Arts & Humanities) using graph-based machine learning.
Explore the effectiveness of Node Similarity and FastRP embeddings for journal classification.
Analyze challenges posed by small datasets, class imbalance, and multi-dimensional journals in graph-based classification.

Dataset
The dataset is sourced from migration research [1] and includes:

Journals: 126 nodes with properties like Journal_Name and Journal_Publisher.
Papers: 30,441 nodes with properties like Paper_field (e.g., "Sociology;Computer Science").
Relationships:
PUBLISHED_IN: Connects papers to journals.
CITES: Connects papers to other papers they cite.



Data Files
The raw dataset consists of the following CSV files, located in the data/raw/ directory:

authors.csv: Author ID, Author Name, Author URL.
journal.csv: Journal Name, Journal Publisher.
paper.csv: Paper ID, Paper DOI, Paper Title, Paper Year, Paper URL, Paper Citation Count, Field of Study, Journal Volume, Journal Date.
topic.csv: Topic ID, Topic Name, Topic URL.
paper_journal.csv: Paper ID, Journal Name, Journal Publisher.
paper_topic.csv: Paper ID, Topic ID.
paper_reference.csv: Paper ID, Referenced Paper ID.

These files are cleaned and preprocessed using the Data_cleaning.ipynb notebook, producing cleaned versions in the data/cleaned/ directory (e.g., cleaned_author_data.csv, cleaned_journal_data.csv).
Preprocessing
The preprocessing steps include:

Removing duplicates and null values.
Normalizing fields (e.g., trimming spaces, handling missing publishers with "Not Published").
Converting numeric fields (e.g., Paper Year, Paper Citation Count) and dates (Journal Date) to appropriate types.
Splitting combined paper fields (e.g., "Sociology;Computer Science" → ["Sociology", "Computer Science"]) and mapping to five categories:
Social Sciences: Sociology, Political Science, Economics, Psychology, History, Business.
Natural Sciences: Geography (natural sciences context), Mathematics, Physics, Chemistry, Geology.
Life Sciences & Medicine: Medicine, Biology, Environmental Science.
Engineering & Technology: Engineering, Computer Science, Materials Science.
Arts & Humanities: Art, Philosophy.



Row counts before and after cleaning:

Cleaned_author_data.csv: Before = 38,926, After = 38,854
Cleaned_author_paper.csv: Before = 56,451, After = 56,451
Cleaned_journal_data.csv: Before = 166, After = 127
Cleaned_paper.csv: Before = 693,753, After = 30,442
Cleaned_paper_data_valid_rows.csv: Before = 693,753, After = 30,442
Cleaned_paper_journal.csv: Before = 32,648, After = 32,648
Cleaned_topic.csv: Before = 6,490, After = 6,490

Prerequisites
To run this project, you need the following software and tools:

Neo4j: Version compatible with GDS 2.13.4 (e.g., Neo4j Desktop or Community Edition 4.x). Download from Neo4j.
Neo4j Graph Data Science (GDS): Version 2.13.4. Install via Neo4j Desktop or manually from Neo4j GDS.
Python: Version 3.8 or higher. Download from Python.
Python Libraries:
graphdatascience: For interacting with Neo4j GDS.
pandas: For data preprocessing.
notebook: For running Jupyter notebooks.
numpy, scikit-learn: For FastRP model training and evaluation.


Jupyter Notebook: For executing the data cleaning script.

Install the required Python libraries using:
pip install graphdatascience pandas notebook numpy scikit-learn

Ensure the Neo4j database is running and accessible at bolt://localhost:7687 with the default credentials (username: neo4j, password: 12345678) or your configured credentials.
Setup Instructions
Follow these steps to set up the project environment and prepare the data:

Clone the Repository:
git clone https://github.com/syedfahadnadeem082/Gds_Project.git
cd Gds_Project


Set Up Neo4j:

Install Neo4j and start the database.
Install the GDS plugin (version 2.13.4) via Neo4j Desktop or by copying the plugin JAR to the Neo4j plugins directory.
Configure Neo4j to allow file imports by setting dbms.directories.import=import in the Neo4j configuration file.
Create a database named neo4j and ensure it’s accessible at bolt://localhost:7687.


Data Cleaning:

Open Data_cleaning.ipynb in Jupyter Notebook.
Run all cells to clean the raw CSV files in data/raw/ and produce cleaned versions in data/cleaned/.
Ensure the output files (e.g., cleaned_author_data.csv, cleaned_journal_data.csv) are generated correctly.


Load Data into Neo4j:

Place the cleaned CSV files in the Neo4j import directory (e.g., <neo4j-home>/import).
Execute the Cypher queries in load_data.cypher to load nodes (Author, Paper, Journal, Topic) and relationships (AUTHORED, CITES, HAS_TOPIC, PUBLISHED_IN). Example query for loading journals:CALL apoc.periodic.iterate(
"LOAD CSV WITH HEADERS FROM 'file:///cleaned_journal_data.csv' AS row RETURN row",
"MERGE (j:Journal {Journal_Name: row.`Journal Name`, Journal_Publisher: row.`Journal Publisher`})",
{batchSize: 2000}
);


Run similar queries for other nodes and relationships (see load_data.cypher for details).


Assign CategoryIds:

Run the category assignment query to set categoryId on Journal nodes based on paper fields:MATCH (j:Journal)<-[:PUBLISHED_IN]-(p:Paper)
WHERE p.Paper_field IS NOT NULL
WITH j, collect(DISTINCT p.Paper_field) AS fields
WITH j, apoc.coll.flatten([field IN fields | split(field, ';')]) AS split_fields
WITH j, [field IN split_fields | trim(field)] AS trimmed_fields
WITH j, apoc.coll.toSet([field IN trimmed_fields |
    CASE field
        WHEN field IN ['Sociology', 'Political Science', 'Economics', 'Psychology', 'History', 'Business'] THEN 'Social Sciences'
        WHEN field IN ['Geography', 'Mathematics', 'Physics', 'Chemistry', 'Geology'] THEN 'Natural Sciences'
        WHEN field IN ['Medicine', 'Biology', 'Environmental Science'] THEN 'Life Sciences & Medicine'
        WHEN field IN ['Engineering', 'Computer Science', 'Materials Science'] THEN 'Engineering & Technology'
        WHEN field IN ['Art', 'Philosophy'] THEN 'Arts & Humanities'
        ELSE NULL
    END
]) AS unique_cats
SET j.category =
    CASE
        WHEN size(unique_cats) = 0 THEN 'Interdisciplinary'
        WHEN size(unique_cats) = 1 THEN unique_cats[0]
        ELSE reduce(s = '', cat IN unique_cats | s + cat + '/')
    END;


Fix trailing slashes:MATCH (j:Journal)
WHERE j.category IS NOT NULL AND toString(j.category) ENDS WITH '/'
SET j.category = left(toString(j.category), size(toString(j.category)) - 1);





Running the Project
Node Similarity Approach

Ensure the Neo4j database is running with the loaded data.
Run the node_similarity.py script:python scripts/node_similarity.py


The script performs:
Dropping existing SIMILAR_TO relationships.
Projecting the graph (30,563 nodes, 98,343 relationships).
Deduplicating journals (empty result in one run).
Computing SIMILAR_TO relationships using Jaccard similarity (interrupted in one run).
Splitting data into 85% train and 15% test sets.
Computing node similarity and predicting categoryIds for test journals (completed in a later run).



FastRP Approach

Ensure the Neo4j database is running with the loaded data.
Run the fast_rp.py script or fast_rp_pipeline.ipynb notebook (check the scripts/ directory for the exact file).
Follow the instructions to:
Project the graph for FastRP (30,567 nodes, 111,283 relationships).
Generate 1024-dimensional embeddings with weights [0.7, 0.2, 0.1].
Train a logistic regression model with a 15% test split, 3 validation folds, and class weights [1.0, 1.5, 1.2, 1.8, 1.6, 2.5, 2.0].
Evaluate the model to obtain accuracy and F1 score.



Results
FastRP Approach

Test Accuracy: 31.6%
F1_MACRO: 6.9%
All predictions were category 0 (Social Sciences), indicating challenges with class imbalance and the small dataset.

Node Similarity Approach

The approach was implemented but faced performance issues. One run stopped at computing SIMILAR_TO relationships (Step 5), producing a graph projection with 30,563 nodes and 98,343 relationships in 134 ms. A later run completed through prediction (Step 11), but no evaluation was conducted due to computational inefficiencies and sparse similarity matrices.

Project Structure

data/:
raw/: Raw CSV files (e.g., authors.csv, journal.csv).
cleaned/: Cleaned CSV files (e.g., cleaned_author_data.csv).


scripts/:
Data_cleaning.ipynb: Notebook for cleaning raw data.
load_data.cypher: Cypher queries to load data into Neo4j.
node_similarity.py: Script for Node Similarity approach.
fast_rp.py or fast_rp_pipeline.ipynb: Script/notebook for FastRP approach.


results/: Output files or figures from model evaluations (if generated).

Challenges and Future Work
Challenges

Small Dataset: Limited to 126 journals, restricting model training.
Class Imbalance: Category 0 (Social Sciences, 37 journals) dominated.
Multi-Dimensional Journals: Diverse paper fields (e.g., Sociology and Computer Science) complicated classification.
GDS Limitations: Logistic regression in GDS 2.13.4 struggled with imbalanced data.

Future Work

Increase Dataset Size: Include more journals to improve model performance.
Advanced Models: Use Graph Neural Networks (GNNs) in newer GDS versions.
Additional Features: Incorporate paper citation counts or metadata.
Paper Clustering: Cluster papers to infer categories or use hybrid classification.

References

Rothenberger, L., Pasta, M. Q., & Mayerhoffer, D. (2021). "Mapping and impact assessment of phenomenon-oriented research fields: The example of migration research." Quantitative Science Studies, 2(4), 1466–1485. https://doi.org/10.1162/qss_a_00163

