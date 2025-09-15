
## DO with OpenAI also
from langchain_ollama import ChatOllama
from pydantic import BaseModel,Field
from typing import TypedDict,Optional,Literal

model = ChatOllama(
    model="llama3"
)
#json_schema
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_things": {
      "title": "Key Things",
      "type": "array",
      "items": { "type": "string" },
      "description": "write down the key themes discussed in the review"
    },
    "summary": {
      "title": "Summary",
      "type": "array",
      "items": { "type": "string" },
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "title": "Sentiment",
      "enum": ["pos", "neg"],
      "type": "string",
      "description": "return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "title": "Pros",
      "type": "array",
      "items": { "type": "string" },
      "description": "write down all the pros inside the list"
    },
    "cons": {
      "title": "Cons",
      "type": "array",
      "items": { "type": "string" },
      "description": "write down all the cons inside the list"
    }
  },
  "required": ["key_things", "summary", "sentiment"]
}
st_model = model.with_structured_output(json_schema)
## Try with big reviews
res = st_model.invoke('''The samsung Phone has decent all-day battery life, but it drains fast under heavy use and seems to lose efficiency over time''')

print(res)