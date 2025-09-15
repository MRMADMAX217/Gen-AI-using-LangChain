# from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
# from dotenv import load_dotenv
# from sklearn.metrics.pairwise import cosine_similarity

# load_dotenv()
# embedding=OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)
# document=[
#     "Monu is cse 1"
#     "Dil is cse 6"
#     "sidh is cse 1"
# ]
# query1="Sidh is handsome"
# query2="Dil is handsome"
# doc_embeddings=embedding.embed_documents(document)
# query1_embeddings=embedding.embed_query(query1)
# query2_embeddings=embedding.embed_query(query2)

# print(cosine_similarity([query1_embeddings],document))
# print(cosine_similarity([query2_embeddings],document))


from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Load Hugging Face embedding model
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Documents
documents = [
    "Monu is cse 1",
    "Dil is cse 6",
    "Sidh is cse 1",
    "tanooj is ece 4"
]

# Queries
query1 = "cse 1 is handsome"
query2 = "Dil is handsome"

# Get embeddings
doc_embeddings = embedding.embed_documents(documents)
query1_embedding = embedding.embed_query(query1)
query2_embedding = embedding.embed_query(query2)

# Cosine similarity
print("Query1 vs Documents:", cosine_similarity([query1_embedding], doc_embeddings))
print("Query2 vs Documents:", cosine_similarity([query2_embedding], doc_embeddings))
