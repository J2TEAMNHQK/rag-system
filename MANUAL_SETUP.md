# Manual Setup Guide - RAG System

## Step-by-Step Instructions

### 1. Open Command Prompt or PowerShell
```bash
cd "C:\Users\Admin\Documents\Folder demo"
```

### 2. Activate Virtual Environment
```bash
rag_env\Scripts\activate
```

### 3. Verify Installation
```bash
python -c "import torch; import gradio; print('Dependencies OK')"
```

### 4. Run the System
Try these options in order:

**Option A - Optimized Version:**
```bash
python main_working.py
```

**Option B - Full Version:**
```bash
python main_simplified.py
```

**Option C - Original Version (if dependencies complete):**
```bash
python main.py
```

### 5. Access the System
Open your browser and go to:
- http://localhost:7860
- http://127.0.0.1:7860
- If those fail, check the console output for alternative URLs

### 6. Troubleshooting

**If port 7860 is busy:**
```bash
netstat -ano | findstr 7860
```

**Try different port:**
Edit the Python file and change `server_port=7860` to `server_port=8080`

**If models take too long to download:**
The system will download AI models on first run. This can take 5-15 minutes depending on internet speed.

**If you see API errors:**
- Check Windows Firewall settings
- Try running as Administrator
- Check antivirus software

### 7. Test the System
Once running:
1. Type a question about your documents in the text box
2. Click "Send"
3. The system will search through your TXT files and provide answers

### 8. Current Status
✅ Virtual environment created
✅ Dependencies installed
✅ Documents loaded (1.txt, 2.txt)
✅ Code files ready
🔄 System launching (in progress)

### 9. Manual Launch (If Automated Launch Fails)

Create a simple test file:
```python
# test_simple.py
import gradio as gr

def echo(message, history):
    return history + [[message, f"Echo: {message}"]]

demo = gr.ChatInterface(
    fn=echo,
    title="Test Interface"
)

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
```

Run: `python test_simple.py`

If this works, the issue is with the RAG components, not Gradio itself.

### 10. Expected Behavior
When working correctly, you should see:
```
Starting Simple RAG System...
Initializing RAG system...
Loaded: 1.txt
Loaded: 2.txt
Processed X chunks from 2 documents
Loading embedding model...
RAG system ready!
Launching interface...
* Running on local URL: http://127.0.0.1:7860
```

Then open the URL in your browser.