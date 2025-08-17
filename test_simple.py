import gradio as gr
import os

def simple_rag(message, history):
    """Simple document search without complex AI models"""
    
    # Read documents
    docs_dir = "./documents"
    found_content = []
    
    if os.path.exists(docs_dir):
        for filename in os.listdir(docs_dir):
            if filename.endswith('.txt'):
                try:
                    with open(os.path.join(docs_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Simple keyword search
                        if any(word.lower() in content.lower() for word in message.split()):
                            # Get first 500 characters that contain the keywords
                            lines = content.split('\n')
                            relevant_lines = []
                            for line in lines:
                                if any(word.lower() in line.lower() for word in message.split()):
                                    relevant_lines.append(line)
                                if len(relevant_lines) >= 3:
                                    break
                            
                            if relevant_lines:
                                found_content.append(f"From {filename}:\n" + "\n".join(relevant_lines[:3]))
                except Exception as e:
                    found_content.append(f"Error reading {filename}: {e}")
    
    if found_content:
        response = f"I found information related to your question:\n\n" + "\n\n".join(found_content)
    else:
        response = f"I searched through the documents but couldn't find specific information related to: {message}"
    
    history.append([message, response])
    return history

# Create interface
with gr.Blocks(title="Simple Document Search") as demo:
    gr.Markdown("# Simple Document Search")
    gr.Markdown("This is a basic keyword-based search through your documents.")
    
    chatbot = gr.Chatbot(
        label="Document Search Results",
        height=500
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Search Query",
            placeholder="Enter keywords to search in documents...",
            scale=4
        )
        send_btn = gr.Button("Search", scale=1)
    
    clear_btn = gr.Button("Clear")
    
    msg.submit(simple_rag, inputs=[msg, chatbot], outputs=[chatbot]).then(
        lambda: "", outputs=[msg]
    )
    
    send_btn.click(simple_rag, inputs=[msg, chatbot], outputs=[chatbot]).then(
        lambda: "", outputs=[msg]
    )
    
    clear_btn.click(lambda: [], outputs=[chatbot])

if __name__ == "__main__":
    print("Starting Simple Document Search...")
    print("This version uses basic keyword matching instead of AI models")
    print("It should start much faster!")
    
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=True
        )
    except Exception as e:
        print(f"Launch failed: {e}")
        try:
            demo.launch(server_port=7861, share=False)
        except Exception as e2:
            print(f"Alternative port failed: {e2}")
            print("Please try manual launch or check network settings")