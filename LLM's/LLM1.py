
import ollama

res = ollama.chat(model='llama3',messages=[
{'role': 'user', 'content': 'Hello, are you '}]
                  )


print(res['message']['content'])