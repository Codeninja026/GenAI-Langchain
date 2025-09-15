
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal

model = ChatOllama(
    model="llama3"
)
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['pos','neg'] = Field(description="Give the sentiment of the feedback")


parser2 = PydanticOutputParser(pydantic_object=Feedback)
prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback into positive or negative. \n"
             "Respond ONLY with valid JSON.\n"
             "{text}. {format_instruction}",
    input_variables=['text'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)


clf_chain = prompt1 | model | parser2


prompt2= PromptTemplate(
    template="write a  response on the positive feedback .\n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="write a  response on the negative feedback .\n {feedback}",
    input_variables=['feedback']
)


branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='pos',prompt2 | model | parser),
    (lambda x:x.sentiment=='neg',prompt3 | model | parser),
    RunnableLambda(
        lambda x:"could not find sentiment"
    )
)

chain = clf_chain | branch_chain

final_res = chain.invoke({'text':'This is a terrible smartphone'})



print(final_res)

chain.get_graph().print_ascii()