
"""
Harry Potter Gen Z - Retrieval Augmented Generation (RAG) System

This script implements a RAG system for Harry Potter content with Gen Z language.
It loads documents, splits them into chunks, creates embeddings, and stores them in a vector database.
"""
import os
import shutil
from typing import List, Optional
from dotenv import load_dotenv
import time
import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Langchain imports
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

# Gemini imports
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")


if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
genai.configure(api_key=GEMINI_API_KEY)
print("Gemini API configured successfully")

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "book_1"  # Default to book_1, can be changed to "data/books" or other directories
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MODEL_NAME = "gemini-1.5-flash-latest"  # Default model
EMBEDDING_MODEL = "models/embedding-001"  # Gemini embedding model

class HarryPotterRAG:
    """RAG system for Harry Potter content"""
    print("Initializing RAG system...niko hapa!")
    def __init__(self, data_path: str = DATA_PATH, chroma_path: str = CHROMA_PATH):
        """Initialize the RAG system"""
        self.data_path = data_path
        self.chroma_path = chroma_path
        print(f"Embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
        self.vectorstore = None

    def load_documents(self) -> List[Document]:
        """Load documents from the data directory"""
        loader = DirectoryLoader(self.data_path, glob="**/*.md")
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from {self.data_path}")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks: List[Document], recreate: bool = False) -> None:
        """Create a vector store from document chunks"""
        # Clear existing database if requested
        if recreate and os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)
            print(f"Removed existing database at {self.chroma_path}")

        # Create and persist the vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.chroma_path
        )
        self.vectorstore.persist()
        print(f"Created and persisted vector store with {len(chunks)} chunks")

    def load_vectorstore(self) -> None:
        """Load an existing vector store"""
        if not os.path.exists(self.chroma_path):
            raise ValueError(f"Vector store not found at {self.chroma_path}")

        self.vectorstore = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embedding_model
        )
        print(f"Loaded existing vector store from {self.chroma_path}")

    def query(self, query_text: str, k: int = 3) -> dict:
        """Query the RAG system"""
        if not self.vectorstore:
            try:
                self.load_vectorstore()
            except ValueError:
                raise ValueError("Vector store not initialized. Run create_index() first.")

        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(query_text, k=k)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # Create prompt
        prompt = ChatPromptTemplate.from_template("""
            Yo bestie, you’re a **savage AI assistant** fluent in **Gen Z lingo + brain rot vibes** 😤✨.  
            Your mission? Drop 🔥 answers to the question below, but ONLY based on the **magical context** provided.

            Context (keep it 100 w/ this info ONLY): 

        {context}

        ---

        Question: {question}

       Speak full **Gen Z mode**: slang, chaos, and good vibes ✅  
        Sprinkle in **Harry Potter refs** where it slaps 🪄🐍  
        Don’t be shy with those **emojis** (but don’t overkill 👀)  
        Keep it **sassy but helpful**, and make sure it’s **understandable af**.

        🚨 No making stuff up outside the context, fr. Keep it canon. 🚨
        """)

        # Generate response
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": query_text})

        return {
            "query": query_text,
            "response": response.content,
            "source_documents": docs
        }

    def create_index(self, recreate: bool = False) -> None:
        """Create the complete index pipeline"""
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        self.create_vectorstore(chunks, recreate=recreate)
        print("Index creation complete!")
        
    def generate_response(payload: dict) -> dict:
        form = payload.get('formValue', {})
        prompt = form.get('prompt', '')
        tone = form.get('tone', 'neutral')
        include_sources = form.get('includeSources', False)

        # Example logic
        response_text = f"Prompt: '{prompt}' (Tone: {tone})"
        if include_sources:
            response_text += "\nSources will be included."

        return {
            "response": response_text,
            "status": "success"
        }

def main(payload: dict):
    """Main function to demonstrate RAG functionality"""
    print("🧙‍♂️ Harry Potter Gen Z - Retrieval Augmented Generation System 🧙‍♂️")

    # Initialize RAG system
    rag = HarryPotterRAG()
    print('====================>', payload)

    form = payload.get('formValue', {})
    
    print('====================>', form)

    print("Form value:", form)
    prompt = payload.get('prompt', '')
    tone = payload.get('tone', 'neutral')
    include_sources = payload.get('includeSources', False)

    response_text = f"Prompt: '{prompt}' (Tone: {tone})"
    
    print('====================>')
    
    print(f"Prompt: {response_text}")
    

    # Create index if it doesn't exist
    if not os.path.exists(CHROMA_PATH):
        print("Creating new vector index...")
        rag.create_index()
    else:
        print("Using existing vector index...")
        rag.load_vectorstore()

    # Interactive query loop
    print("\n✨ Ready to answer your Harry Potter questions in Gen Z style! ✨")
    print("Type 'exit' to quit")

    if prompt.lower() in ["exit", "quit", "q"]:
        return {"status": "aborted", "message": "User exited."}

    try:
        start_time = time.time()
        result = rag.query(prompt)
        end_time = time.time()

        response_data = {
            "response": result["response"],
            "time_taken": f"{end_time - start_time:.2f}s",
            "sources": []
        }

        if include_sources and "source_documents" in result:
            response_data["sources"] = [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "content": doc.page_content[:200]
                }
                for doc in result["source_documents"]
            ]

        return response_data

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    main()



