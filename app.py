import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain.chains import RetrievalQA


os.environ["GOOGLE_API_KEY"] = "paste_your_key_here"


loader = PyPDFLoader("test.pdf") 
docs = loader.load() 
print(f"Loaded {len(docs)} pages"print("Splitting...")
splitter = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=200 ) 
chunks = splitter.split_documents(docs) 
print(f"Created {len(chunks)} chunks")


print("Storing in ChromaDB...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
vectorstore = Chroma.from_documents(chunks, embeddings) 
print("Stored!")

print("Setting up QA chain...") 
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") 
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())


question = "What is this document about?" 
print(f"\nQuestion: {question}") 
answer = qa_chain.invoke({"query": question}) 
print(f"Answer: {answer['result']}")
