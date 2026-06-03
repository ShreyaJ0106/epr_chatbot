import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_vector_db():

    print("Loading PDFs...")

    documents = []

    PDF_FOLDER = "epr_docs"

    for file in os.listdir(PDF_FOLDER):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(
                PDF_FOLDER,
                file
            )

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            for doc in docs:

                doc.metadata["source"] = file

            documents.extend(docs)

    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating ChromaDB...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    print("ChromaDB Created Successfully")


if __name__ == "__main__":

    create_vector_db()