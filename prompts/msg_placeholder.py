
## llm not  inherited here so you can add
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
chat_temp = ChatPromptTemplate(
    [
        ('system','your are helpful customer agent'),
        MessagesPlaceholder(variable_name='chat_hist'),
        ('human', '{query}')
    ]
)
chat_hist = []
with open('prev_chat_msg_placeholder.txt') as file:
    for line in file:
        role, content = line.strip().split("|", 1)
        if role.lower() == "human":
            chat_hist.append(HumanMessage(content=content))
        elif role.lower() == "ai":
            chat_hist.append(AIMessage(content=content))

print(chat_hist)

prompt = chat_temp.invoke({'chat_hist':chat_hist,
                  'query':'where is my refund'})

print(prompt)