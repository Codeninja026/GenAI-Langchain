from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

model = ChatOllama(
    model = 'llama3'
)

schema = [
    ResponseSchema(name='fact1',description="Fact 1 about the topic"),
    ResponseSchema(name='fact2',description="Fact 1 about the topic"),
    ResponseSchema(name='fact3',description="Fact 1 about the topic")

]

parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template="Give 3 facts about the {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables= {'format_instruction':parser.get_format_instructions()}
)


print(template.format(topic = 'google'))
print(template.invoke({'topic':'google'}))