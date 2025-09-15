from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableBranch,RunnableLambda,RunnablePassthrough
from langchain.prompts import PromptTemplate

model = ChatOllama(
    model='llama3'
)

prompt = PromptTemplate(
    template="Write email on the topic {topic}",
    input_variables=['topic']
)
prompt_summarizer = PromptTemplate(
    template="Summarize this email in 10 words.{topic}",
    input_variables=['topic']
)
parser = StrOutputParser()
chain_seq = RunnableSequence(prompt,model,parser)

def no_of_words(text):
    if len(text.split())>10:
        return True
    else:
        return False

chain1 = RunnableBranch(
    (no_of_words,
     RunnableSequence(prompt_summarizer,model,parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(chain_seq,chain1)
res = chain.invoke({'topic':'cricket'})
print(res)
