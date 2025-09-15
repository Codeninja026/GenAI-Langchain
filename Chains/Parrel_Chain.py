
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# used to execute multiple chains
from langchain.schema.runnable import RunnableParallel
model = ChatOllama(
    model="llama3"
)


template1 = PromptTemplate(
    template="Generate a short notes of the following text. \n"
             "{text}" ,
    input_variables=['text']
)

template2 =  PromptTemplate(
    template="Generate 5 questions from the following text. \n"
             "{text}",
    input_variables=['text']
)


template3 = PromptTemplate(
    template="merge the provided notes and quiz into a single document .\n"
             "notes -> {notes} , quiz -> {quiz}",
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chains = RunnableParallel({
    'notes': template1 | model | parser ,
    'quiz': template2 | model | parser
})

merge_chain = template3 | model | parser

chain  = parallel_chains | merge_chain
## text on transformer
text ='''A Transformer refers to a neural network architecture primarily used in large language models (LLMs) to process and generate text by focusing on word relationships through an attention mechanism, rather than sequential processing. This allows for efficient handling of long-range dependencies in text, enabling tasks like translation, summarization, and text generation. The term can also refer to a power transformer, an electrical device that changes voltage in alternating current circuits. 
AI Transformer Models (for Text)
Purpose:
To understand and generate human-like text by identifying complex relationships between words in a sentence, even if they are far apart. 
Mechanism:
Unlike older models that processed text sequentially, transformers use a self-attention mechanism to weigh the importance of different words in the input text, allowing them to grasp context more effectively. 
Process:
Tokenization: Input text is broken down into smaller units called "tokens" (words or sub-words). 
Embeddings: Tokens are converted into high-dimensional vectors (embeddings) that capture their semantic meaning. 
Attention: The attention mechanism adjusts these embeddings to encode contextual meaning, considering how tokens relate to each other. 
Output: The model then generates text one token at a time, predicting the next word based on the processed input. 
Applications:
These models are the foundation for various natural language processing (NLP) tasks, including text summarization, translation, text classification, and question answering. '''
res= chain.invoke({'text':text})

print(res)

chain.get_graph().print_ascii()