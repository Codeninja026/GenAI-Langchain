
## DO with OpenAI also
from langchain_ollama import ChatOllama
from pydantic import BaseModel,Field
from typing import TypedDict,Optional,Literal

model = ChatOllama(
    model="llama3"
)

class Review(BaseModel):
    key_things: list[str] = Field(description="write down the key themes discussed in the review")
    summary:list[str]=Field(description=("A brief summary of the review"))
    sentiment:Literal['pos','neg']=Field("return sentiment of the review either negative,positive or neutral")
    pros : Optional[list[str]]=Field(description="write down all the pros inside the list")
    cons : Optional[list[str]]=Field(description="write down all the cons inside the list")

st_model = model.with_structured_output(Review)


## Try with big reviews
res = st_model.invoke('''The samsung Phone has decent all-day battery life, but it drains fast under heavy use and seems to lose efficiency over time''')

print(res)