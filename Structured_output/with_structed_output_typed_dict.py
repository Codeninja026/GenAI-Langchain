## DO with OpenAI
from langchain_ollama import ChatOllama
from typing import TypedDict,Annotated,Optional,Literal

model = ChatOllama(
    model="llama3"
)

class Review(TypedDict):
    key_things: Annotated[str,"write down the key themes discussed in the review"]
    summary:Annotated[str,"A brief summary of the review"]
    sentiment:Annotated[Literal['pos','neg'],"return sentiment of the review either negative,positive or neutral"]
    pros : Annotated[Optional[list[str]],"write down all the pros inside the list"]
    cons : Annotated[Optional[list[str]],"write down all the cons inside the list"]

st_model = model.with_structured_output(Review)


## Try with big reviews
res = st_model.invoke('''The samsung Phone has decent all-day battery life, but it drains fast under heavy use and seems to lose efficiency over time''')

print(res)