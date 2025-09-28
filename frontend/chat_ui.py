import gradio as gr
import httpx

API_URL = "http://localhost:8000/chat"

def chat_with_bot(message):
    try:
        response = httpx.post(API_URL, json={"message": message}, timeout=20)
        if response.status_code == 200:
            answer = response.json().get("response", "")
        else:
            answer = f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        answer = f"Error: {str(e)}"
    return answer

with gr.Blocks() as demo:
    gr.Markdown("# RAG bot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message", placeholder="Type your question and press Enter")
    clear = gr.Button("Clear")

    def respond(user_message, chat_history):
        answer = chat_with_bot(user_message, chat_history)
        chat_history = chat_history + [[user_message, answer]]
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
