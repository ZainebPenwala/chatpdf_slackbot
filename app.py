# import re
# import PyPDF2
# from flask import Flask, render_template, request
# from werkzeug.utils import secure_filename
# import re
# import PyPDF2
# import os
# import langchain
# from langchain.document_loaders import PyPDFLoader
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI


# app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         # Handle form submission
#         query = request.form['query']
#         pdf = request.files['pdf']
        
#         # Save the uploaded PDF to a local file
#         pdf_path = 'uploads/' + secure_filename(pdf.filename)
#         pdf.save(pdf_path)
        
#         # Load the uploaded PDF and extract code block text
#         pdf_file = open(pdf_path, 'rb')
#         pdf_reader = PyPDF2.PdfReader(pdf_file)
#         text_within_code_blocks = []
        
#         for page_num in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_num]
#             page_text = page.extract_text()
            
#             # Use regular expressions to find the text within code blocks
#             json_blocks = re.findall(r'{.*?}', page_text, re.DOTALL)
#             text_within_code_blocks.extend(json_blocks)
        
#         # Create the OpenAI language model instance
#         llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
        
#         # Create the document loader and retrieve the pages from the PDF
#         loader = PyPDFLoader(pdf_path)
#         pages = loader.load_and_split()
        
#         # Create the vector store and database
#         db = FAISS.from_documents(pages, OpenAIEmbeddings())
#         retriever = db.as_retriever()
        
#         # Create the retrieval-based QA system
#         qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        
#         result = qa({"query": query})
#         return render_template('index.html', query=query, result=result['result'])
#     else:
#         # Display the initial form
#         return render_template('index.html')

# if __name__ == '__main__':
#     app.run()



import re
import PyPDF2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import langchain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Handle form submission
        query = request.form['query']
        pdf = request.files['pdf']
        
        # Save the uploaded PDF to a local file
        pdf_path = 'uploads/' + secure_filename(pdf.filename)
        pdf.save(pdf_path)
        
        # Create the OpenAI language model instance
        llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
        
        # Create the document loader and retrieve the pages from the PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        
        # Create the vector store and database
        db = FAISS.from_documents(pages, OpenAIEmbeddings())
        retriever = db.as_retriever()
        
        # Create the retrieval-based QA system
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        
        result = qa({"query": query})
        return render_template('index.html', query=query, result=result['result'])
    else:
        # Display the initial form
        return render_template('index.html')
        # return render_template('index.html', query=query, result=result['result'])
    
    


if __name__ == '__main__':
    app.run(debug=True)


