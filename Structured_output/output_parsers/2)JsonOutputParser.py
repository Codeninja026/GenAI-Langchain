from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pandas.core.dtypes.cast import can_hold_element

model = ChatOllama(
    model='llama3'
)

parser = JsonOutputParser()

template = PromptTemplate(
    template='what is the name,age ceo of google \n {format_instruction}',
    input_variables=[],
    # before run time it is filled so it is called partial_variable
    partial_variables= {'format_instruction':parser.get_format_instructions()}

)
#without chain
# prompt = template.format()
#
# res = model.invoke(prompt)
#
# final_res = parser.parse(res.content)
# print(final_res)

#with chain

chain  = template | model | parser

final_res = chain.invoke({})
print(final_res)


