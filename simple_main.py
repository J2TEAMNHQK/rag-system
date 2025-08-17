import os
import sys

print("Starting simple RAG system test...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Check if documents exist
docs_dir = "./documents"
if os.path.exists(docs_dir):
    files = [f for f in os.listdir(docs_dir) if f.endswith('.txt')]
    print(f"Found {len(files)} text files: {files}")
    
    # Read first file as test
    if files:
        with open(os.path.join(docs_dir, files[0]), 'r', encoding='utf-8') as f:
            content = f.read()[:200]
            print(f"Sample content from {files[0]}: {content}...")
else:
    print("Documents directory not found")

print("Test completed successfully!")
print("To run the full RAG system, make sure all dependencies are properly installed.")
print("Then run: python main.py")
print("The system will be available at: http://127.0.0.1:7860")