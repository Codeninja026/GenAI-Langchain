
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(
    model="llama3"
)

template = PromptTemplate(
    template="write a detailed report on the topic {topic}",
    input_variables=['topic']
)
template2= PromptTemplate(
    template="write a 5 line summary of the {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template | model | parser | template2 | model | parser

res = chain.invoke({'topic':'cricket'})

print(res)
chain.get_graph().print_ascii()