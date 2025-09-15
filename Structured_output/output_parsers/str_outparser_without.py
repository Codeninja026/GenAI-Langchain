
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

model = ChatOllama(
    model="llama3"
)

template1 = PromptTemplate(
    template='write a detail report on the {topic}',
    input_variables=['topic']
)


template2 = PromptTemplate(
    template='write a 4 line summary on the following {text} ./n',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'blackhole'})

res = model.invoke(prompt1)


prompt2 = template2.invoke({'text':res.content})

res2 = model.invoke(prompt2)

print(res2.content)