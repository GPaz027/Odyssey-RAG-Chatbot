import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

PERSIST_DIRECTORY = os.getenv('PERSIST_DIRECTORY_TEXT')
COLLECTION_NAME = os.getenv('CHROMADB_COLLECTION_TEXT')
OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4", temperature=0)

def generate_embeddings(splitted_documents):
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY): # If the folder exists and it is not empty
        print("Loading existing Chroma collection...")
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=OpenAIEmbeddings()
        )
    else:
        print("Creating new Chroma collection...")
        vector_store = Chroma.from_documents(
            documents=splitted_documents,
            embedding=OpenAIEmbeddings(),
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
    return vector_store

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)



class TextChain:
    def __init__(self):
        loader = TextLoader("documents/odyssey.txt", encoding="utf-8")
        documents = loader.load()
        chunk_size = 1000
        chunk_overlap = 200
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splitted_documents = text_splitter.split_documents(documents) 
        self.vector_store = generate_embeddings(splitted_documents)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        PROMPT = """
        You are an assistant chatbot that is an expert on Homer's Odyssey. You will answer the user's questions ONLY based on the received context,  
        DO NOT use external sources. In your answers, be precise and answer in a concise way (2-5 sentences).

        - If context does not contain the answer, just say "I don't know".
        - In your answer, include useful metadata if present in context.
                    
        Context: 
        {context}

        User question: 
        {question}
        """

        self.prompt = ChatPromptTemplate.from_template(PROMPT)
        
        # Chain
        self.rag_chain = (
            {"context": self.retriever | format_docs, 
             "question": RunnablePassthrough()
            } 
            | self.prompt
            | llm 
            | StrOutputParser()
        )
