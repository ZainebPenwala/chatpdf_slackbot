## READ THE CODE BLOCKS FROM PDF

# import re
# import PyPDF2

# pdf_file = open('docs/api.pdf', 'rb')
# pdf_reader = PyPDF2.PdfReader(pdf_file)

# text_within_code_blocks = []

# for page_num in range(len(pdf_reader.pages)):
#     page = pdf_reader.pages[page_num]
#     page_text = page.extract_text()
#     # print(page_text)

#     # Use regular expressions to find the text within code blocks
#     json_blocks = re.findall(r'{.*?}', page_text, re.DOTALL)
#     text_within_code_blocks.extend(json_blocks)

# print(text_within_code_blocks)


import re
import PyPDF2
import os
import langchain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = ''

# OPENAI_API_KEY = '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'

# print()
# print("TEMPERATURE == 0.1")
# print()

## creates an instance of the OpenAI language model
llm = OpenAI(model_name="gpt-3.5-turbo",  temperature=0.5)

loader = PyPDFLoader("docs/api.pdf")
pages = loader.load_and_split()

# Use PyPDF2 to extract text within code blocks
pdf_file = open('docs/api.pdf', 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)

text_within_code_blocks = []

for page_num in range(len(pdf_reader.pages)):
    page = pdf_reader.pages[page_num]
    page_text = page.extract_text()

    # Use regular expressions to find the text within code blocks
    json_blocks = re.findall(r'{.*?}', page_text, re.DOTALL)
    text_within_code_blocks.extend(json_blocks)

# print(text_within_code_blocks)

db = FAISS.from_documents(pages, OpenAIEmbeddings())

retriever = db.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True)

# i = 0
# while i == 0:
while True:
    query = input("please add a search query: ")
    result = qa({"query": query})

    print()
    print("Question: ", query)
    print()
    print("Answer: ", result['result'])
    print()
