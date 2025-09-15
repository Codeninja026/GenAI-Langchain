
from langchain_community.document_loaders import CSVLoader


loader = CSVLoader(file_path="college_database_enrollments.csv")

docs = loader.load()

print(docs[0])