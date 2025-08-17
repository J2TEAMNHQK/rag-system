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

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import gradio as gr

# ==========================================
# Configuration
# ==========================================

class SystemConfig:
    EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    LLM_MODEL = "microsoft/DialoGPT-medium"
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 5
    TOP_K_RERANK = 3
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
            print("Please add your TXT files to this directory and restart the application")
            return documents_data
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
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
                if len(chunk.strip()) > 50:
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
# Simple Vector Database
# ==========================================

class SimpleVectorDatabase:
    def __init__(self):
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': config.DEVICE}
        )
        self.documents = []
        self.vectors = []
    
    def build_database(self, documents):
        print("Building vector database...")
        self.documents = documents
        
        # Create embeddings for all documents
        texts = [doc.page_content for doc in documents]
        self.vectors = self.embeddings.embed_documents(texts)
        
        print(f"Vector database ready with {len(documents)} documents")
    
    def similarity_search(self, query, k=5):
        if not self.vectors:
            return []
        
        # Get query embedding
        query_vector = self.embeddings.embed_query(query)
        
        # Calculate cosine similarities
        similarities = []
        for i, doc_vector in enumerate(self.vectors):
            # Calculate cosine similarity
            dot_product = np.dot(query_vector, doc_vector)
            norm_query = np.linalg.norm(query_vector)
            norm_doc = np.linalg.norm(doc_vector)
            similarity = dot_product / (norm_query * norm_doc)
            similarities.append((similarity, i))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        results = []
        for sim, idx in similarities[:k]:
            results.append((self.documents[idx], 1 - sim))  # Convert to distance
        
        return results

# ==========================================
# BM25 Search
# ==========================================

class BM25Search:
    def __init__(self):
        self.bm25 = None
        self.documents = []
    
    def setup(self, documents):
        print("Setting up BM25 search...")
        self.documents = documents
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        print("BM25 search ready")
    
    def search(self, query, k=config.TOP_K_RETRIEVAL):
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'document': self.documents[idx],
                    'score': float(scores[idx])
                })
        return results

# ==========================================
# Hybrid Retriever
# ==========================================

class HybridRetriever:
    def __init__(self, vector_db, bm25_search):
        self.vector_db = vector_db
        self.bm25_search = bm25_search
        
        print("Loading reranking model...")
        self.reranker = SentenceTransformer(
            config.RERANK_MODEL,
            device=config.DEVICE
        )
        print("Reranking model ready")
    
    def retrieve(self, query):
        vector_results = self.vector_db.similarity_search(query, k=config.TOP_K_RETRIEVAL)
        bm25_results = self.bm25_search.search(query)
        
        combined_docs = {}
        
        for doc, score in vector_results:
            doc_id = doc.page_content
            combined_docs[doc_id] = {
                'document': doc,
                'vector_score': 1 - score,
                'bm25_score': 0
            }
        
        for result in bm25_results:
            doc_id = result['document'].page_content
            if doc_id in combined_docs:
                combined_docs[doc_id]['bm25_score'] = result['score']
            else:
                combined_docs[doc_id] = {
                    'document': result['document'],
                    'vector_score': 0,
                    'bm25_score': result['score']
                }
        
        for doc_id in combined_docs:
            vector_score = combined_docs[doc_id]['vector_score']
            bm25_score = min(combined_docs[doc_id]['bm25_score'] / 10, 1.0)
            combined_docs[doc_id]['hybrid_score'] = 0.7 * vector_score + 0.3 * bm25_score
        
        sorted_docs = sorted(
            combined_docs.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )[:config.TOP_K_RETRIEVAL]
        
        return self.rerank(query, sorted_docs)
    
    def rerank(self, query, results):
        if len(results) <= 1:
            return [r['document'] for r in results[:config.TOP_K_RERANK]]
        
        pairs = [(query, result['document'].page_content) for result in results]
        rerank_scores = self.reranker.predict(pairs)
        
        for i, result in enumerate(results):
            result['rerank_score'] = float(rerank_scores[i])
        
        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        return [r['document'] for r in reranked[:config.TOP_K_RERANK]]

# ==========================================
# Simple LLM
# ==========================================

class SimpleLLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        print("Loading language model...")
        
        model_name = "microsoft/DialoGPT-medium"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if config.DEVICE == "cuda" else torch.float32
        )
        
        if config.DEVICE == "cuda":
            self.model = self.model.to(config.DEVICE)
        
        print("Language model loaded successfully")
    
    def generate(self, prompt, max_new_tokens=256):
        inputs = self.tokenizer.encode(
            prompt, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True
        )
        
        if config.DEVICE == "cuda":
            inputs = inputs.to(config.DEVICE)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
        return generated_text if generated_text else "I need more context to provide a helpful answer."

# ==========================================
# Simple RAG System
# ==========================================

class SimpleRAGSystem:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the context below, answer the question accurately and concisely:

Context:
{context}

Question: {question}

Answer:"""
        )
    
    def answer_question(self, question):
        docs = self.retriever.retrieve(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = self.qa_prompt.format(context=context, question=question)
        answer = self.llm.generate(prompt)
        
        sources = []
        for doc in docs[:3]:
            sources.append({
                'content': doc.page_content[:200] + "...",
                'source': doc.metadata.get('source', 'Unknown')
            })
        
        return answer, sources

# ==========================================
# Initialize System
# ==========================================

def initialize_rag_system():
    print(f"Initializing RAG system...")
    print(f"Device: {config.DEVICE}")
    
    # Load documents
    doc_processor = DocumentProcessor()
    documents_data = doc_processor.load_documents_from_directory(config.DOCUMENTS_DIR)
    
    if not documents_data:
        print("No documents found. Please add TXT files to the documents directory.")
        return None
    
    processed_documents = doc_processor.process_documents(documents_data)
    
    # Setup vector database
    vector_db = SimpleVectorDatabase()
    vector_db.build_database(processed_documents)
    
    # Setup BM25
    bm25_search = BM25Search()
    bm25_search.setup(processed_documents)
    
    # Setup hybrid retriever
    hybrid_retriever = HybridRetriever(vector_db, bm25_search)
    
    # Setup LLM
    simple_llm = SimpleLLM()
    
    # Setup RAG system
    rag_system = SimpleRAGSystem(simple_llm, hybrid_retriever)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("RAG system ready")
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
                response += "\n\nNguon tham khao:\n"
                for i, source in enumerate(sources, 1):
                    response += f"{i}. {source['source']}: {source['content']}\n"
            
            history.append([message, response])
            return history
        
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            history.append([message, error_msg])
            return history
    
    with gr.Blocks(title="RAG System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Simple RAG System")
        gr.Markdown("Ask questions about documents in the system")
        
        chatbot = gr.Chatbot(
            label="AI Assistant",
            height=500,
            show_copy_button=True
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your question",
                placeholder="Enter your question...",
                scale=4
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
        
        clear_btn = gr.Button("Clear chat history")
        
        msg.submit(
            chat_interface,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "",
            outputs=[msg]
        )
        
        send_btn.click(
            chat_interface,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "",
            outputs=[msg]
        )
        
        clear_btn.click(
            lambda: [],
            outputs=[chatbot]
        )
    
    return demo

# ==========================================
# Main Function
# ==========================================

def main():
    print("Starting Simple RAG System...")
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    
    if rag_system is None:
        print("Failed to initialize RAG system. Please check your documents directory.")
        return
    
    # Create and launch Gradio interface
    demo = create_gradio_interface(rag_system)
    
    print("Launching Gradio interface...")
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            quiet=False
        )
    except Exception as e:
        print(f"Gradio launch failed: {e}")
        print("Trying alternative launch method...")
        try:
            demo.launch(
                share=False,
                debug=False,
                server_port=7861,
                show_error=True
            )
        except Exception as e2:
            print(f"Alternative launch also failed: {e2}")
            print("Please try running manually or check network settings.")

if __name__ == "__main__":
    main()