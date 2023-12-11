Copy code
from elasticsearch import Elasticsearch

# Connect to Elasticsearch
# es = Elasticsearch('localhost', port=9200)

# # Example query to find similar vectors (adjust according to your setup)
# query = {
#     "query": {
#         "script_score": {
#             "query": {"match_all": {}},
#             "script": {
#                 "source": "cosineSimilarity(params.query_vector, 'embedding_field') + 1.0",
#                 "params": {
#                     "query_vector": [0.1, 0.5, 0.9]  # Replace with your query vector
#                 }
#             }
#         }
#     }
# }

# # Execute the query
# results = es.search(index='your_index_name', body=query)

################################################################################
from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch('localhost', port=9200)

# Example vectors (replace with your actual vectors)
vectors = [
    [0.1, 0.2, 0.3],  # Vector 1
    [0.4, 0.5, 0.6],  # Vector 2
    # Add more vectors here
]

# Example index creation (if index doesn't exist)
index_name = 'vector_index'

if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body={"mappings": {"properties": {"vector": {"type": "dense_vector"}}}})

# Upload vectors to Elasticsearch
for i, vector in enumerate(vectors):
    es.index(index=index_name, body={"vector": vector, "id": i + 1})
