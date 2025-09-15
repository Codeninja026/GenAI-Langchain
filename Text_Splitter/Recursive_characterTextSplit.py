
from langchain.text_splitter import RecursiveCharacterTextSplitter


text = """
my name is llucky
Iam 17 years old

I live in warangl
how are you """

splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(chunks)