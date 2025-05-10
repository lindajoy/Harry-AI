import os
import shutil
from typing import List
import nltk

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from .config import (
    GEMINI_API_KEY, CHROMA_PATH, DATA_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, MODEL_NAME, EMBEDDING_MODEL
)

# Download NLTK once at module load
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Configure Gemini
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
genai.configure(api_key=GEMINI_API_KEY)
print("✅ Gemini API configured")


class HarryPotterRAG:
    """Retrieval-Augmented Generation system for Harry Potter content in Gen Z language."""

    def __init__(self, data_path: str = DATA_PATH, chroma_path: str = CHROMA_PATH):
        self.data_path = data_path
        self.chroma_path = chroma_path
        print(f"🛠️ Initializing RAG with embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        self.llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
        self.vectorstore = None

    def load_documents(self) -> List[Document]:
        """Loads Markdown documents from the data directory."""
        loader = DirectoryLoader(self.data_path, glob="**/*.md")
        documents = loader.load()
        print(f"📄 Loaded {len(documents)} documents")
        for doc in documents:  # Preview first
            print(f"📝 {doc.metadata['source']}")
        return documents


    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits documents into overlapping text chunks for vector storage."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"🧩 Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks: List[Document], recreate: bool = False) -> None:
        """Builds a vector store (Chroma DB) from document chunks."""
        if recreate and os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)
            print(f"🧹 Existing vector store at {self.chroma_path} cleared")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.chroma_path
        )
        self.vectorstore.persist()
        print(f"✅ Vector store created with {len(chunks)} chunks")

    def load_vectorstore(self) -> None:
        """Loads an existing vector store from disk."""
        if not os.path.exists(self.chroma_path):
            raise ValueError(f"Vector store not found at {self.chroma_path}")

        self.vectorstore = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embedding_model
        )
        print(f"📦 Vector store loaded from {self.chroma_path}")

    def query(self, query_text: str, k: int = 3):
        """Executes a query and streams the Gen Z styled response."""
        if not self.vectorstore:
            self.load_vectorstore()

        docs = self.vectorstore.similarity_search(query_text, k=k)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_template("""
        Yo bestie, you’re a **savage AI assistant** fluent in **Gen Z lingo + brain rot vibes** 😤✨.
        Your mission? Drop 🔥 answers to the question below, but ONLY based on the **magical context** provided.

        Context: {context}

        💭 **Question:** {question}

        📝 **Your Answer:**
        - Gen Z slang + Harry Potter vibes
        - Keep it 🔥 and on point
        """)

        chain = prompt | self.llm
        stream = chain.stream({"context": context, "question": query_text})

        for chunk in stream:
            content = getattr(chunk, "content", None)
            if content:
                yield content
        yield {"source_documents": docs}

    def build_index_pipeline(self, recreate: bool = False) -> None:
        """Full pipeline: load → split → create vectorstore."""
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        self.create_vectorstore(chunks, recreate=recreate)
        print("🏗️ Index build complete")

# class DocumentLoader:
#     def __init__(self, data_path: str):
#         self.data_path = data_path
#
#     def load_documents(self) -> List[Document]:
#         """Loads Markdown documents from the data directory."""
#         loader = DirectoryLoader(self.data_path, glob="**/*.md")
#         documents = loader.load()
#         print(f"📄 Loaded {len(documents)} documents")
#         return documents
#
# def main():
#     DATA_PATH = "../data"
#     loader = DocumentLoader(DATA_PATH)
#     documents = loader.load_documents()
#     # You can process the documents here if needed
#     for doc in documents:  # Preview first 3
#         print(f"📝 {doc.metadata['source']}")
#
# if __name__ == "__main__":
#     main()