from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

def no_of_words(text):
    return len(text.split())

model = ChatOllama(
    model="llama3"
)
prompt1 = PromptTemplate(
    template="Write a joke about the topic {topic}.(in 5 words only) ",
    input_variables=['topic']
)
parser  =StrOutputParser()
joke_gen_chain = RunnableSequence(prompt1,model,parser)


par_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'no_of_words': RunnableLambda(no_of_words)
})

chain = RunnableSequence(joke_gen_chain,par_chain)



res= chain.invoke({'topic':'cricket'})
print(res)