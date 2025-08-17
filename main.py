import os
import gc
import torch
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import gradio as gr

# ==========================================
# Configuration
# ==========================================

class SystemConfig:
    EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    LLM_MODEL = "microsoft/DialoGPT-medium"  # Placeholder for Llama 3.2
    
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
# Vector Database
# ==========================================

class VectorDatabase:
    def __init__(self):
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': config.DEVICE}
        )
        self.vector_db = None
    
    def build_database(self, documents):
        print("Building vector database...")
        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        print(f"Vector database ready with {len(documents)} documents")
    
    def load_existing_database(self):
        if os.path.exists("./chroma_db"):
            print("Loading existing vector database...")
            self.vector_db = Chroma(
                embedding_function=self.embeddings,
                persist_directory="./chroma_db"
            )
            return True
        return False

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
        vector_results = self.vector_db.vector_db.similarity_search_with_score(
            query, k=config.TOP_K_RETRIEVAL
        )
        
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
# Llama 3.2 LLM
# ==========================================

class Llama32LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        print("Loading Llama 3.2 model...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        model_name = "microsoft/DialoGPT-medium"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        print("Llama 3.2 model loaded successfully")
    
    def generate(self, prompt, max_new_tokens=256):
        inputs = self.tokenizer.encode(
            prompt, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True
        )
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
        return generated_text

# ==========================================
# LangChain Wrapper
# ==========================================

from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

class LlamaLLMWrapper(LLM):
    llama_model: Any
    
    def __init__(self, llama_instance):
        super().__init__()
        self.llama_model = llama_instance
    
    @property
    def _llm_type(self) -> str:
        return "llama_3_2"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self.llama_model.generate(prompt)
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": "llama_3_2"}

# ==========================================
# RAG System
# ==========================================

class AdvancedRAGSystem:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        
        self.rewrite_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Rewrite this query to be more specific for document search:
            Original: {query}
            Rewritten:"""
        )
        
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the context below, answer the question accurately and concisely:

Context:
{context}

Question: {question}

Answer:"""
        )
    
    def rewrite_query(self, query):
        try:
            prompt = self.rewrite_prompt.format(query=query)
            rewritten = self.llm._call(prompt)
            return rewritten.strip() if rewritten.strip() else query
        except:
            return query
    
    def answer_question(self, question):
        rewritten_query = self.rewrite_query(question)
        docs = self.retriever.retrieve(rewritten_query)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = self.qa_prompt.format(context=context, question=question)
        answer = self.llm._call(prompt)
        
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
        return None
    
    processed_documents = doc_processor.process_documents(documents_data)
    
    # Setup vector database
    vector_db = VectorDatabase()
    if not vector_db.load_existing_database():
        vector_db.build_database(processed_documents)
    
    # Setup BM25
    bm25_search = BM25Search()
    bm25_search.setup(processed_documents)
    
    # Setup hybrid retriever
    hybrid_retriever = HybridRetriever(vector_db, bm25_search)
    
    # Setup LLM
    llama_llm = Llama32LLM()
    llm_wrapper = LlamaLLMWrapper(llama_llm)
    
    # Setup RAG system
    rag_system = AdvancedRAGSystem(llm_wrapper, hybrid_retriever)
    
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
        gr.Markdown("# RAG System with Llama 3.2")
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
    print("Starting RAG System...")
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    
    # Create and launch Gradio interface
    demo = create_gradio_interface(rag_system)
    
    print("Launching Gradio interface...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=False
    )

if __name__ == "__main__":
    main()