import os
from datetime import datetime

import pandas as pd
import gradio as gr

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ==================================
# LOAD FAQ DATASET
# ==================================

print("Loading FAQ Dataset...")

faq_df = pd.read_csv("data/epr_faq.csv")

# ==================================
# LOAD EMBEDDINGS
# ==================================

print("Loading Embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==================================
# LOAD CHROMADB
# ==================================

print("Loading ChromaDB...")

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

print("Ready!")

# ==================================
# HISTORY SETUP
# ==================================

os.makedirs("history", exist_ok=True)

history_file = "history/chat_history.csv"

if not os.path.exists(history_file):

    pd.DataFrame(
        columns=[
            "timestamp",
            "question",
            "answer"
        ]
    ).to_csv(
        history_file,
        index=False
    )


# ==================================
# SAVE HISTORY
# ==================================

def save_history(question, answer):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    new_row = pd.DataFrame(
        [[timestamp, question, answer]],
        columns=[
            "timestamp",
            "question",
            "answer"
        ]
    )

    new_row.to_csv(
        history_file,
        mode="a",
        header=False,
        index=False
    )


# ==================================
# LOAD HISTORY
# ==================================

def load_history():

    if os.path.exists(history_file):

        return pd.read_csv(history_file)

    return pd.DataFrame(
        columns=[
            "timestamp",
            "question",
            "answer"
        ]
    )


# ==================================
# FAQ SEARCH
# ==================================

def search_faq(question):

    question = question.lower().strip()

    # Exact Match

    for _, row in faq_df.iterrows():

        faq_question = str(
            row["question"]
        ).lower().strip()

        if faq_question == question:

            return row["answer"]

    # Partial Match

    for _, row in faq_df.iterrows():

        faq_question = str(
            row["question"]
        ).lower().strip()

        if question in faq_question:

            return row["answer"]

    return None


# ==================================
# CHATBOT
# ==================================

def epr_chat(message, history):

    # FAQ SEARCH

    faq_answer = search_faq(message)

    if faq_answer:

        save_history(
            message,
            faq_answer
        )

        return faq_answer

    # PDF SEARCH

    try:

        results = db.similarity_search_with_score(
            message,
            k=1
        )

        if not results:

            response = (
                "No relevant information found."
            )

            save_history(
                message,
                response
            )

            return response

        doc, score = results[0]

        print(
            "Similarity Score:",
            score
        )

        # Reject weak matches

        if score > 1.0:

            response = (
                "No relevant information found in uploaded EPR documents."
            )

            save_history(
                message,
                response
            )

            return response

        answer = doc.page_content

        save_history(
            message,
            answer
        )

        return answer

    except Exception as e:

        error_message = (
            f"Error: {str(e)}"
        )

        save_history(
            message,
            error_message
        )

        return error_message


# ==================================
# THEME
# ==================================

theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="slate"
)

css = """
.gradio-container {
    background: #F4FFF7;
}

h1 {
    text-align: center;
    color: #1B5E20;
}

body {
    background-color: #F4FFF7;
}
"""


# ==================================
# UI
# ==================================

with gr.Blocks(
    theme=theme,
    css=css,
    title="EPR Compliance Assistant"
) as demo:

    gr.Markdown("""
# ♻️ EPR Compliance Assistant

### Empowering Sustainable Compliance Through Intelligent EPR Insights

---
""")

    gr.ChatInterface(
        fn=epr_chat
    )

    gr.Markdown("## 📜 Chat History")

    history_table = gr.Dataframe(
        label="Saved Conversations"
    )

    refresh_btn = gr.Button(
        "🔄 Refresh History"
    )

    refresh_btn.click(
        fn=load_history,
        outputs=history_table
    )

    gr.File(
        value=history_file,
        label="📥 Download Chat History"
    )

demo.launch()