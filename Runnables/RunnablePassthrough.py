from langchain_ollama import ChatOllama
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser

from Chains.Parrel_Chain import parallel_chains

model = ChatOllama(
    model="llama3"
)

prompt1 = PromptTemplate(
    template="Write a joke on the topic {topic} (5 words).",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Write the explanation of the joke {joke}(6 words).",
    input_variables=['joke']
)
parser = StructuredOutputParser()

chain1 = RunnableSequence(prompt1,model,parser)

parallel_chain =RunnableParallel({
     'joke':RunnablePassthrough(),
      'joke_exp':RunnableSequence(prompt2,model,parser)
})

chain = RunnableSequence(chain1,parallel_chain)


res = chain.invoke({'topic':'cricket'})
print(res)