import os
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import GROQ_API_KEY, LLM_MODEL, CHROMA_DIR, DATA_FILE
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
def load_split_documents():
    loader = TextLoader(DATA_FILE)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(documents)

def setup_vectorstore(docs):
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
        vectorstore.persist()

    return vectorstore

def build_rag_chain():
    docs = load_split_documents()
    vectorstore = setup_vectorstore(docs)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "lambda_mult": 0.5}
    )

    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=512
    )

    prompt = PromptTemplate(
        template=(
            "You are a precise and reliable AI assistant tasked with answering questions strictly based on the given context.\n"
            "Do NOT use prior knowledge or make assumptions beyond the context.\n"
            "If the answer is not explicitly stated or inferable from the context, respond with: "
            "'I'm sorry, the information is not available in the provided context.'\n"
            "Always prioritize factual accuracy, clarity, and transparency in your responses.\n\n"
            "------------------------\n"
            "Context:\n{context}\n"
            "------------------------\n"
            "Question: {question}\n"
            "Answer:"
        ),
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return chain
