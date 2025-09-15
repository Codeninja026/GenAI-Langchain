
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

model = ChatOllama(
    model='llama3'
)

class Person(BaseModel):
    name : str = Field(description="name of the person")
    age : int = Field(gt=18,description="age of the person")
    city : str = Field(description="Name of the city the person belongs to ")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="what is the name,age,city ceo of {company} \n {format_instruction}",
    input_variables=['company'],
    # before run time it is filled so it is called partial_variable
    partial_variables= {'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

res = chain.invoke({'company':'google'})
print(res)