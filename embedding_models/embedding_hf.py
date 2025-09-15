
from langchain_huggingface import  HuggingFaceEndpointEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
api  = os.getenv('HUGGINGFACEHUB_API_TOKEN')

embeddings = HuggingFaceEndpointEmbeddings(
    model='sentence-transformers/all-MiniLM-L6-v2',
    huggingfacehub_api_token=api)


res = embeddings.embed_query("hyd is capital of telangana")
print(str(res))
print(type(res))
print(res)