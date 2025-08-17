import os
import gc
import torch
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import gradio as gr

# ==========================================
# Configuration
# ==========================================

class SystemConfig:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Smaller, faster model
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    TOP_K_RETRIEVAL = 3
    DEVICE = "cpu"  # Force CPU for stability
    DOCUMENTS_DIR = "./documents"

config = SystemConfig()

# ==========================================
# Document Processing
# ==========================================

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
    
    def load_documents_from_directory(self, directory_path):
        documents_data = []
        
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")
            return documents_data
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        documents_data.append({
                            'filename': filename,
                            'content': content
                        })
                    print(f"Loaded: {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        return documents_data
    
    def process_documents(self, documents_data):
        documents = []
        for doc_data in documents_data:
            chunks = self.text_splitter.split_text(doc_data['content'])
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 30:
                    documents.append(Document(
                        page_content=chunk.strip(),
                        metadata={
                            'source': doc_data['filename'],
                            'chunk_id': i
                        }
                    ))
        
        print(f"Processed {len(documents)} chunks from {len(documents_data)} documents")
        return documents

# ==========================================
# Simple RAG System
# ==========================================

class SimpleRAGSystem:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.vectors = []
        self.bm25 = None
        
    def initialize(self, documents):
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': config.DEVICE}
        )
        
        self.documents = documents
        
        print("Creating embeddings...")
        texts = [doc.page_content for doc in documents]
        self.vectors = self.embeddings.embed_documents(texts)
        
        print("Setting up BM25...")
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        print("RAG system ready!")
        
    def search(self, query, k=3):
        if not self.vectors or not self.bm25:
            return []
        
        # Vector search
        query_vector = self.embeddings.embed_query(query)
        similarities = []
        for i, doc_vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, doc_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
            )
            similarities.append((similarity, i))
        
        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Combine scores
        combined_scores = []
        for i, (sim, idx) in enumerate(similarities):
            bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0
            combined_score = 0.7 * sim + 0.3 * min(bm25_score / 10, 1.0)
            combined_scores.append((combined_score, idx))
        
        # Sort and return top k
        combined_scores.sort(reverse=True)
        results = []
        for score, idx in combined_scores[:k]:
            results.append(self.documents[idx])
        
        return results
    
    def answer_question(self, question):
        docs = self.search(question)
        if not docs:
            return "Sorry, I couldn't find relevant information to answer your question.", []
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Simple answer based on context
        answer = f"Based on the documents, here's what I found:\n\n{context[:1000]}..."
        
        sources = []
        for doc in docs:
            sources.append({
                'content': doc.page_content[:200] + "...",
                'source': doc.metadata.get('source', 'Unknown')
            })
        
        return answer, sources

# ==========================================
# Initialize System
# ==========================================

def initialize_rag_system():
    print("Initializing RAG system...")
    
    # Load documents
    doc_processor = DocumentProcessor()
    documents_data = doc_processor.load_documents_from_directory(config.DOCUMENTS_DIR)
    
    if not documents_data:
        print("No documents found. Please add TXT files to the documents directory.")
        return None
    
    processed_documents = doc_processor.process_documents(documents_data)
    
    if not processed_documents:
        print("No valid document chunks found.")
        return None
    
    # Initialize RAG system
    rag_system = SimpleRAGSystem()
    rag_system.initialize(processed_documents)
    
    return rag_system

# ==========================================
# Gradio Interface
# ==========================================

def create_gradio_interface(rag_system):
    def chat_interface(message, history):
        if not message.strip():
            return history
        
        if rag_system is None:
            error_msg = "System not initialized. Please add documents to ./documents directory and restart."
            history.append([message, error_msg])
            return history
        
        try:
            answer, sources = rag_system.answer_question(message)
            
            response = answer
            if sources:
                response += "\n\nSources:\n"
                for i, source in enumerate(sources, 1):
                    response += f"{i}. {source['source']}: {source['content']}\n"
            
            history.append([message, response])
            return history
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history.append([message, error_msg])
            return history
    
    with gr.Blocks(title="Simple RAG", theme=gr.themes.Default()) as demo:
        gr.Markdown("# Simple RAG System")
        gr.Markdown("Ask questions about your documents")
        
        chatbot = gr.Chatbot(
            label="Assistant",
            height=400
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Question",
                placeholder="Type your question here...",
                scale=4
            )
            send_btn = gr.Button("Send", scale=1)
        
        clear_btn = gr.Button("Clear")
        
        def send_message(message, history):
            return chat_interface(message, history)
        
        def clear_history():
            return []
        
        msg.submit(send_message, inputs=[msg, chatbot], outputs=[chatbot]).then(
            lambda: "", outputs=[msg]
        )
        
        send_btn.click(send_message, inputs=[msg, chatbot], outputs=[chatbot]).then(
            lambda: "", outputs=[msg]
        )
        
        clear_btn.click(clear_history, outputs=[chatbot])
    
    return demo

# ==========================================
# Main Function
# ==========================================

def main():
    print("Starting Simple RAG System...")
    
    try:
        # Initialize RAG system
        rag_system = initialize_rag_system()
        
        if rag_system is None:
            print("Failed to initialize. Please check documents directory.")
            return
        
        # Create interface
        demo = create_gradio_interface(rag_system)
        
        print("Launching interface...")
        print("If this fails, you can run manually by:")
        print("1. Opening browser")
        print("2. Going to http://localhost:7860")
        print("3. Or trying http://127.0.0.1:7860")
        
        # Try multiple launch configurations
        try:
            demo.launch(
                server_name="localhost",
                server_port=7860,
                share=False,
                inbrowser=True,
                show_error=True
            )
        except Exception as e1:
            print(f"First launch failed: {e1}")
            try:
                demo.launch(
                    server_port=7861,
                    share=False,
                    inbrowser=True
                )
            except Exception as e2:
                print(f"Second launch failed: {e2}")
                try:
                    demo.launch(share=True, show_error=True)
                    print("Launched with share=True - check the public URL above")
                except Exception as e3:
                    print(f"All launch attempts failed: {e3}")
                    print("Manual steps:")
                    print("1. Check if port 7860 is available")
                    print("2. Try different port with: demo.launch(server_port=8080)")
                    print("3. Check firewall settings")
    
    except Exception as e:
        print(f"System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()