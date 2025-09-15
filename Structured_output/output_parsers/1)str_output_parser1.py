
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatOllama(
    model = "llama3"
)


template1 = PromptTemplate(
    template='write a detail report on the {topic}',
    input_variables=['topic']
)


template2 = PromptTemplate(
    template='write a 4 line summary on the following {text} ./n',
    input_variables=['text']
)

parser = StrOutputParser()

# pipeline
chain = template1 | model | parser | template2 | model | parser

res = chain.invoke({'topic':'blackhole'})
print(res)