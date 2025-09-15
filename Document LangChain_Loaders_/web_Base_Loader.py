from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
model = ChatOllama(
    model="llama3"
)
parser = StrOutputParser()

template = PromptTemplate(
    template='''what is name of the movie having the highest imdb.
    on the following text.\n {text}''',
    input_variables=['text']
)
url = 'https://www.imdb.com/chart/top/'
loader = WebBaseLoader(
    url
)

doc = loader.load()
data = doc[0].page_content
chain = template | model | parser

res = chain.invoke({'text': data})

print(res)

#Output after run
# According to the text, the movie
# with the highest IMDb rating is:
#
# 1. The Shawshank Redemption (1994) - R9.3 (3.1M)
#
# Note: The IMDb rating is based on a
# formula that takes into account both the
# number of ratings and the value of those
# ratings from regular users.
