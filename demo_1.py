import openai
import pandas as pd
import numpy as np
import os
from getpass import getpass
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
from langchain.llms import OpenAI
import langchain
from langchain.document_loaders import PyPDFLoader
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA



# os.environ['OPENAI_API_KEY'] = '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'

openai.api_key = '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'
os.environ['OPENAI_API_KEY'] = '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'


df = pd.read_csv('words.csv')
# print(df)


## TO CREATE WORD EMBEDDINGS
# df['embedding'] = df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
# df.to_csv('word_embeddings.csv')
print("word embeddings created")

## READ THE EMBEDDINGS
df = pd.read_csv('word_embeddings.csv')
df['embedding'] = df['embedding'].apply(eval).apply(np.array)
# print(df)

search_term = input('Enter a search term: ')

## CREATE EMBEDDINGS FOR SERACH TERMS
search_term_vector = get_embedding(search_term, engine="text-embedding-ada-002")
# print(search_term_vector)

## CALCULATE COSINE SIMILARITY
df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))

## SORT BY SIMILARITY
df.sort_values("similarities", ascending=False).head(20)

# print(df)

## Filter rows based on condition
filtered_df = df[df['similarities'] >= 0.8]
filtered_df['similarities'] = filtered_df['similarities'].astype(float)

sorted_df = filtered_df.sort_values(by='similarities', ascending=False)

print()
print(sorted_df)
print()

matching_terms_list = sorted_df['text'].tolist()

print(matching_terms_list)
print()

docs_list = ['buger', 'pizza', 'quesadilla', 'sandwhich', 'zotero']

search_term = 'demo'
doc_path = ''

for term in matching_terms_list:
    if term in docs_list:
        doc_path = 'docs/' + term + '.pdf'
        search_term = term
        break
    else:
        doc_path = 'docs/DEMO.pdf'

    
print(doc_path)
## query the pdf

llm = OpenAI(model_name="gpt-3.5-turbo")

loader = PyPDFLoader(doc_path)
# print(doc_path)
pages = loader.load_and_split()

db = FAISS.from_documents(pages, OpenAIEmbeddings())

retriever = db.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True)

# query = f"what is {search_term} and where was it originated from?"

query = f"what is {search_term} and why is it mentioned so many times??"

result = qa({"query": query})


print()
print("Question: ",query)
print()
print("Answer: ",result['result'])
print()

