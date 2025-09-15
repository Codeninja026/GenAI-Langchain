
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(
    model="llama3"
)

template = PromptTemplate(
    template="write 5 key features of {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = template | model | parser

# res = chain.invoke({'topic':'google'})

# print(type(res))
# print(res)

#output
# <class 'str'>
# Here are five key features of Google:
# 1. **Search Engine**: Google's most iconic feature is its search engine, which allows users to find information on the internet by typing in keywords or phrases. The search engine uses algorithms and natural language processing techniques to retrieve relevant results from the vast amount of data available online.
# 2. **Algorithmic Ranking**: Google's search engine is known for its algorithmic ranking system, which determines the relevance and quality of search results. This algorithm takes into account various factors such as page authority, content freshness, and user engagement to deliver the most useful and relevant results to users.
# 3. **Gmail**: Gmail is a popular email service provided by Google that allows users to send and receive emails with attachments up to 25MB in size. Features like spam filtering, virus protection, and integration with other Google services like Google Drive make it a comprehensive email solution.
# 4. **Google Maps**: Google Maps is a mapping service that provides turn-by-turn directions, street views, and real-time traffic updates for users. It also offers features like Street View, which allows users to explore streets and places remotely, and Local Search, which enables users to find businesses and services in their area.
# 5. **AdWords and AdSense**: Google's advertising platforms, AdWords and AdSense, allow businesses to create targeted ads that appear on the Google search engine and other websites across the internet. AdWords is used by advertisers to create and manage their ads, while AdSense enables website publishers to monetize their content with relevant ads.
# These features have made Google one of the most influential and successful companies in the world, revolutionizing the way people access information, communicate, navigate, and do business online.

chain.get_graph().print_ascii()