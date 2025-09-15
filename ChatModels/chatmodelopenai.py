from langchain_ollama import ChatOllama

# Initialize Ollama LLaMA3 model
llm = ChatOllama(model="llama3",max_completion_tokens=5)

res = llm.invoke("hello this is lucky")

print(res.content)