import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter



PERSIST_DIRECTORY = os.getenv('PERSIST_DIRECTORY')
COLLECTION_NAME = os.getenv('CHROMADB_COLLECTION')
OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4", temperature=0)

def generate_embeddings(splited_documents):
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
            documents=splited_documents,
            embedding=OpenAIEmbeddings(),
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
    return vector_store

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

class PythonDocumentationChain:

    def __init__(self):
        
        # Load Quickstart Documents
        quickstart_file = UnstructuredHTMLLoader("documents/requests-quickstart.html")
        quickstart_docs = quickstart_file.load()
        
        # Tag quickstart documents
        for doc in quickstart_docs:
            doc.metadata["source"] = "quickstart" 
        
        # Load Advanced Documents
        advanced_file = UnstructuredHTMLLoader("documents/requests-advanced.html")
        advanced_docs = advanced_file.load()
        
        # Tag advanced documents
        for doc in advanced_docs:
            doc.metadata["source"] = "advanced" 
        
        # Unify all the quickstart and advanced content
        all_docs = quickstart_docs + advanced_docs

        # Split
        chunk_size = 1000
        chunk_overlap = 200

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splitted_documents = splitter.split_documents(all_docs) 

        # Vector store
        self.vector_store = generate_embeddings(splitted_documents)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        PROMPT = """
        You are an assistant chatbot that is an expert on Requests Python Library. You will answer the user's questions ONLY based on the received context,  
        DO NOT use external sources. In your answers, be precise and answer in a concise way (2-5 sentences).

        - If context does not contain the answer, just say "I don't know".
        - In your answer, mention whether the infromation comes from Quickstart or Advanced.

        Response format:

        - Answer: <YOUR_ANSWER>

        - Source: <<ADVANCED OR QUICKSTART>> 
                    
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




    
