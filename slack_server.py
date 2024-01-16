# import os
# import re
# import PyPDF2
# import glob
# import langchain
# from PyPDF2 import PdfReader
# from slack_sdk import WebClient
# from slack_sdk.errors import SlackApiError
# from slack_bolt.adapter.flask import SlackRequestHandler
# from slack_bolt import App
# from dotenv import find_dotenv, load_dotenv
# from flask import Flask, request
# from langchain.document_loaders import PyPDFLoader
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI


# # Load environment variables from .env file
# load_dotenv(find_dotenv())

# # Set Slack API credentials
# SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
# SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
# SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]

# os.environ['OPENAI_API_KEY'] = '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'

# # Initialize the Slack app
# app = App(token=SLACK_BOT_TOKEN)

# # Initialize the Flask app
# # Flask is a web application framework written in Python
# flask_app = Flask(__name__)
# handler = SlackRequestHandler(app)


# # def buddy_qna(user_input):
# #     llm = ChatOpenAI(model_name="gpt-3.5-turbo",  temperature=0.5)

# #     loader = PyPDFLoader("docs/api.pdf")
# #     pages = loader.load_and_split()

# #     # Use PyPDF2 to extract text within code blocks
# #     pdf_file = open('docs/api.pdf', 'rb')
# #     pdf_reader = PyPDF2.PdfReader(pdf_file)

# #     text_within_code_blocks = []

# #     for page_num in range(len(pdf_reader.pages)):
# #         page = pdf_reader.pages[page_num]
# #         page_text = page.extract_text()

# #         # Use regular expressions to find the text within code blocks
# #         json_blocks = re.findall(r'{.*?}', page_text, re.DOTALL)
# #         text_within_code_blocks.extend(json_blocks)

# #     db = FAISS.from_documents(pages, OpenAIEmbeddings())

# #     retriever = db.as_retriever()

# #     qa = RetrievalQA.from_chain_type(
# #         llm=llm, 
# #         chain_type="stuff", 
# #         retriever=retriever, 
# #         return_source_documents=True)
    
# #     result = qa({"query": user_input})
# #     return result


# def buddy_qna(user_input):
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo",  temperature=0.2)

#     list_of_files = ['cite_1', 'cite_2', 'cite_3', 'cite_4']

#     all_pages = []
#     text_within_code_blocks = []

#     for file in list_of_files:
#         loader = PyPDFLoader(f'docs/citations/{file}.pdf')
#         pages = loader.load_and_split()

#         pdf_file = open(f'docs/citations/{file}.pdf', 'rb')
#         pdf_reader = PyPDF2.PdfReader(pdf_file)

#         text_within_code_blocks = []

#         for page_num in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_num]
#             page_text = page.extract_text()

#             # Use regular expressions to find the text within code blocks
#             json_blocks = re.findall(r'{.*?}', page_text, re.DOTALL)
#             text_within_code_blocks.extend(json_blocks)

#         db = FAISS.from_documents(pages, OpenAIEmbeddings())

#     retriever = db.as_retriever()

#     qa = RetrievalQA.from_chain_type(
#         llm=llm, 
#         chain_type="stuff", 
#         retriever=retriever, 
#         return_source_documents=True)
    
#     result = qa({"query": user_input})
#     return result


# # Decorator for handling direct bot message events
# @app.event("message")
# def handle_direct_message(event, say):
#     if event.get("subtype") is None and event.get("channel_type") == "im":
#         user_input = event["text"]
#         print()
#         print("-------user input = ", user_input)
#         print()

#         # say("Hang on ... I am reading ...")
        
#         result = buddy_qna(user_input)
#         say(result['result'])


# @flask_app.route("/slack/events", methods=["POST"])
# def slack_events():
#     """
#     Route for handling Slack events.
#     This function passes the incoming HTTP request to the SlackRequestHandler for processing.

#     Returns:
#         Response: The result of handling the request.
#     """
#     return handler.handle(request)

# if __name__ == '__main__':
#     flask_app.run(host='127.0.0.1', port=5000, debug=True)




########################### IMPROVEMNET FOR READING CODE BLOCKS AND PPT CONVERTED TO PDF#############3333333

import os
import re
import langchain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]

os.environ['OPENAI_API_KEY'] = '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'

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the Flask app
# Flask is a web application framework written in Python
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)


def buddy_qna(user_input):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    list_of_files = ['cite_1', 'cite_2', 'cite_3', 'cite_4', 'cite_5', 'cite_6']
    
    for file in list_of_files:
        loader = PyPDFLoader(f'docs/citations/{file}.pdf')
        pages = loader.load_and_split()

        text_within_code_blocks = []

        for page in pages:
            page_text = page.page_content

            # Use regular expressions to find the text within code blocks
            json_blocks = re.findall(r'{.*?}', page_text, re.DOTALL)
            text_within_code_blocks.extend(json_blocks)

        db = FAISS.from_documents(pages, OpenAIEmbeddings())

        retriever = db.as_retriever()

        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever, 
            return_source_documents=True
        )
        
        result = qa({"query": user_input})
        # print("Result:", result['result'])
        return result


# Decorator for handling direct bot message events
@app.event("message")
def handle_direct_message(event, say):
    if event.get("subtype") is None and event.get("channel_type") == "im":
        user_input = event["text"]
        print()
        print("-------user input = ", user_input)
        print()

        result = buddy_qna(user_input)
        say(result['result'])


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.

    Returns:
        Response: The result of handling the request.
    """
    return handler.handle(request)


if __name__ == '__main__':
    flask_app.run(host='127.0.0.1', port=5000, debug=True)
