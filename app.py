from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from docx import Document
import pandas as pd

def extract_text_from_pdf(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx):
    doc = Document(docx)
    return " ".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(txt):
    return txt.read().decode()

def extract_text_from_csv(csv):
    df = pd.read_csv(csv)
    return " ".join(df.columns) + " " + " ".join(df.values.flatten().astype(str))

def main():
    load_dotenv()
    st.set_page_config(page_title="cur.io")
    st.image('./logo.png')  # adjust width as per your nee
    user_question = st.text_input("", placeholder="Enter your question here")

    st.sidebar.title("Upload Files")
    uploaded_files = st.sidebar.file_uploader("Upload your files", type=["pdf", "txt", "docx", "csv"], accept_multiple_files=True)

    text = ""
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            text += extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text += extract_text_from_txt(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text += extract_text_from_docx(uploaded_file)
        elif uploaded_file.type == "text/csv": 
            text += extract_text_from_csv(uploaded_file)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    if chunks:
        with st.spinner('Processing...'):  # Add this line
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            if user_question:
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question, max_tokens=500)

                st.write(response)


if __name__ == '__main__':
    main()
