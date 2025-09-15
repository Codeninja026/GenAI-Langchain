
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="cric_info",
    #glob for which type of files we have to pick,all pdf's
    glob="*.pdf",
    loader_cls=PyPDFLoader
)


#           normal loader
# doc = loader.load()


#           Total pages of all pdf
#           1st pdf - pg , 3 pdf - 3 pgs so it returns 3
# print(len(doc))
#           first page
# print(doc[0])

#problem
# But this takes some time if no.of pdf's increase
# we are loading all pdf into ram at a time ,if no.of
# increase there is a chance of memory issues we use Lazy load function here

#            here uses generator object doc by doc load not at once
doc = loader.lazy_load()

for i in doc:
    print(i.metadata)

