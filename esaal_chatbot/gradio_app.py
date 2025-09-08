import gradio as gr
from langchain_orchestrator import orchestrate

chat_history = []

def chatbot_fn(user_input, history):
    history = history or []
    def add_message(role, content):
        history.append({"role": role, "content": content})
    add_message("user", user_input)
    response = orchestrate(user_input)
    add_message("assistant", response)
    return history

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # ESAAL Chatbot Agent 🤖
    Welcome! Ask about the clinic, book appointments, or get help.
    """)
    chatbot = gr.Chatbot(type='messages')
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Type your message and press Enter...")
    clear = gr.Button("Clear Chat")

    def user(user_message, history):
        history = history or []
        history.append({"role": "user", "content": user_message})
        return "", history

    txt.submit(user, [txt, chatbot], [txt, chatbot], queue=False).then(
        chatbot_fn, [txt, chatbot], [chatbot]
    )
    clear.click(lambda: [], None, chatbot, queue=False)

    demo.launch()
