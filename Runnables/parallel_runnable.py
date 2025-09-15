from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import  RunnableParallel,RunnableSequence

prompt1 = PromptTemplate(
    template="write a 5 word tweet of the topic {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="write a 5 keywords of the topic {topic}" ,
    input_variables=['topic']
)
model = ChatOllama(
    model='llama3'
)

parser = StrOutputParser()

chain = RunnableParallel(
    {
            'tweet': RunnableSequence(prompt1,model,parser),
            'keywords':RunnableSequence(prompt2,model,parser )
    }
)

res = chain.invoke({'topic':'cricket'})
print(res)