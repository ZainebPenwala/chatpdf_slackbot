import langchain
from langchain.document_loaders import PyPDFLoader
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


os.environ['OPENAI_API_KEY'] = ''
# getpass.getpass('OpenAI API Key:')

## creates an instance of the OpenAI language model
llm = OpenAI(model_name="gpt-4-1106-preview")

# loader = PyPDFLoader("docs/api.pdf")
loader = PyPDFLoader("/home/zainebpenwala/Documents/projects/mistral_finetune/indiantravelsoft0000unse.pdf")
pages = loader.load_and_split()
# print(pages)

db = FAISS.from_documents(pages, OpenAIEmbeddings())

# docs = faiss_index.similarity_search("How will the community be engaged?", k=2)
# for doc in docs:
#     print(str(doc.metadata["page"]) + ":", doc.page_content[:3000])

retriever = db.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True)

i = 0
while i == 0:
    query = input("please add a search query: ")
    result = qa({"query": query})

    print()
    print("Question: ",query)
    print()
    print("Answer: ",result['result'])
    print()

