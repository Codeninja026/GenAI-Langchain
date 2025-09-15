from langchain_huggingface import HuggingFaceEndpointEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()
api = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=api
)

doc = [
    "Sachin Tendulkar – Legendary Indian batsman, known as the 'God of Cricket' and highest run scorer in international cricket.",
    "Virat Kohli – Modern Indian batting icon, famous for his aggressive playstyle and consistency across formats.",
    "Chris Gayle – West Indian powerhouse, celebrated for his explosive batting in T20 cricket and record-breaking sixes.",
    "AB de Villiers – South African superstar nicknamed 'Mr. 360', admired for his innovative stroke play.",
    "KL Rahul – Stylish Indian batsman and wicketkeeper, known for his versatility across all formats."
]


doc_embedding = embedding.embed_documents(doc)
query = "who is virat kohli?"

query_embedding = embedding.embed_query(query)
## it must be in 2d
values = cosine_similarity([query_embedding],doc_embedding)[0]


idx,score = sorted(list(enumerate(values)),key = lambda x:x[1])[-1]

print(query)
print(doc[idx])
print('similarity score is: ',score)







