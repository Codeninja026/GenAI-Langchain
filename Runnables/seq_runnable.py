from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import  RunnableSequence

prompt = PromptTemplate(
    template="write a 1 line summary of {topic}",
    input_variables=['topic']
)

model = ChatOllama(
    model='llama3'
)

parser = StrOutputParser()

chain = RunnableSequence(
    prompt,model,parser
)

res = chain.invoke({'topic':'cricket'})
print(res)