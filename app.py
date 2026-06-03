import os
import pandas as pd
import gradio as gr

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --------------------------
# CREATE CHROMADB IF MISSING
# --------------------------

if not os.path.exists("chroma_db"):

    print("ChromaDB not found.")

    from ingest import create_vector_db

    create_vector_db()

# --------------------------
# LOAD FAQ
# --------------------------

print("Loading FAQ Dataset...")

faq_df = pd.read_csv(
    "data/epr_faq.csv"
)

# --------------------------
# EMBEDDINGS
# --------------------------

print("Loading Embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------
# LOAD VECTOR DB
# --------------------------

print("Loading ChromaDB...")

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

print("Ready!")

# --------------------------
# FAQ SEARCH
# --------------------------

def search_faq(question):

    question = question.lower().strip()

    for _, row in faq_df.iterrows():

        faq_question = str(
            row["question"]
        ).lower().strip()

        if question == faq_question:

            return row["answer"]

    return None

# --------------------------
# MAIN CHATBOT
# --------------------------

def epr_chat(message, history):

    # FAQ SEARCH

    faq_answer = search_faq(message)

    if faq_answer:

        return faq_answer

    # PDF SEARCH

    docs = db.similarity_search(
        message,
        k=3
    )

    if len(docs) == 0:

        return """
No relevant information found in uploaded EPR documents.
"""

    answer = ""

    for doc in docs:

        answer += doc.page_content

        answer += "\n\n--------------------------\n\n"

    return answer.strip()

# --------------------------
# UI
# --------------------------

css = """
.gradio-container {
    background: #F7FAFC;
}
"""

theme = gr.themes.Soft()

with gr.Blocks(
    theme=theme,
    css=css,
    title="EPR Compliance Assistant"
) as demo:

    gr.Markdown(
        """
# ♻️ EPR Compliance Assistant

### Empowering Sustainable Compliance Through Intelligent EPR Insights
"""
    )

    gr.ChatInterface(
        fn=epr_chat
    )

# --------------------------
# RENDER DEPLOYMENT
# --------------------------

port = int(
    os.environ.get(
        "PORT",
        7860
    )
)

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)