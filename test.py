import os
import re
import PyPDF2
import glob
import langchain
from PyPDF2 import PdfReader
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

os.environ['OPENAI_API_KEY'] = ''

def buddy_qna(user_input):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",  temperature=0.5)

    list_of_files = ['cite_1', 'cite_2', 'cite_3', 'cite_4']

    all_pages = []
    text_within_code_blocks = []

    for file in list_of_files:
        loader = PyPDFLoader(f'docs/citations/{file}.pdf')
        pages = loader.load_and_split()

        pdf_file = open(f'docs/citations/{file}.pdf', 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text_within_code_blocks = []

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()

            # Use regular expressions to find the text within code blocks
            json_blocks = re.findall(r'{.*?}', page_text, re.DOTALL)
            text_within_code_blocks.extend(json_blocks)

        db = FAISS.from_documents(pages, OpenAIEmbeddings())

    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True)
    
    result = qa({"query": user_input})
    return result


user_input = 'What are the different APIs used to generate citations?'
result = buddy_qna(user_input)
print(result['result'])
