import openai
import langchain
from flask import render_template,request,Flask,redirect
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from openai import OpenAI
app = Flask(__name__)

# read document from directory
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

# split document
def chunk_data(docs, chunk_size=500, chunk_overlap=5):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc

embeddings = OpenAIEmbeddings(api_key='your api key')
client =openai.OpenAI(api_key='your api key')


# Load documents
doc = read_doc('documents')
documents = chunk_data(docs=doc)


def create_index():
    
#create index in db
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("faiss_index1")

# create faiss index
db = FAISS.load_local("faiss_index1",embeddings,allow_dangerous_deserialization=True)


@app.route('/')
def main():
    return render_template('index.html')
# res=res, user_input=user_input

# db = FAISS.from_documents(documents,embeddings).save_local("faiss_index")

# vectors=embeddings.embed_query('提供产业链及代表企业分析要点')
# print(vectors)

# while input != "quit()":
#     query = input("User:")
#     docs = db.similarity_search(query)

#     retriever = db.as_retriever()
    
#     docs = retriever.invoke(query)
#     #print(docs[2].page_content)
#     doc_text = documents[0].page_content 
#     completion = client.chat.completions.create(model="gpt-3.5-turbo",
#     messages=[
#     {"role": "system", "content": doc_text},
#     {"role": "user", "content": query},

# ]
# )
#     reply = completion.choices[0].message.content
    
#     print("\n"+ reply + "\n")
# print("your new assistant is ready!")

@app.route('/query', methods=['GET', 'POST'])
def query():
    if request.method == 'GET':
        user_input = str(request.args.get('text'))
        #user_input = request.form['user_input']
        print(user_input)
        
        # Perform similarity search
        docs = db.similarity_search(user_input)
        retriever = db.as_retriever()
        docs = retriever.invoke(user_input)

        # Get document text for OpenAI completion
        doc_text = documents[0].page_content 
        
        # Get completion from OpenAI
        completion = client.chat.completions.create(model="gpt-3.5-turbo",
                                                    messages=[{"role": "system", "content": doc_text},
                                                    {"role": "user", "content": user_input}])

        # Get reply from completion
        res = completion.choices[0].message.content
        return res
        



if __name__ == '__main__':
    app.run(debug=True)


