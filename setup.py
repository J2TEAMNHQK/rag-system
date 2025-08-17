from setuptools import setup, find_packages

setup(
    name="rag-system",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "langchain==0.1.0",
        "langchain-community==0.0.13",
        "chromadb==0.4.24",
        "sentence-transformers==2.2.2",
        "rank-bm25==0.2.2",
        "transformers==4.36.2",
        "torch==2.1.2",
        "gradio==4.16.0",
        "accelerate==0.25.0",
        "bitsandbytes==0.41.3",
    ],
    author="Quang Khai",
    description="Advanced RAG System with Llama 3.2",
    python_requires=">=3.8",
)