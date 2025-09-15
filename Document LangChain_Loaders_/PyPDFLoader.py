
from langchain_community.document_loaders import PyPDFLoader

# can be any page pdf .
# not used for complex layouts
loader = PyPDFLoader("cricket_info.pdf")
doc = loader.load()


print(type(doc))
print(doc[0])
