
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
model = ChatOllama(
    model="llama3"
)

template = PromptTemplate(
    template="Write a summary of the following text./n"
             "{text}.",
    input_variables=['text']
)

loader = TextLoader("cricket.txt",encoding="utf-8")

doc = loader.load()
parser = StrOutputParser()
chain  = template | model |parser

res = chain.invoke({'text':doc[0].page_content})

print(res)