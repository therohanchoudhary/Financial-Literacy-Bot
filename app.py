import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader


def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_file='llama-2-7b-chat.ggmlv3.q4_K_S.bin',
        max_new_tokens=4096,
        temperature=0.6
    )
    return llm


def main():
    st.title("Finance Literacy Bot")

    pdf_folder_path = 'Zerodha Notes'

    loader = PyPDFDirectoryLoader(pdf_folder_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)

    embeddings_mini = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                            model_kwargs={'device': 'cpu'})

    embeddings = embeddings_mini
    docsearch = Chroma.from_documents(documents, embeddings)

    qa = RetrievalQA.from_chain_type(llm=load_llm(), chain_type="stuff", retriever=docsearch.as_retriever())

    query = st.text_input("Enter your question:")
    if st.button("Search"):
        if query:
            result = qa.run(query)
            st.write(f"Answer: {result}")


if __name__ == "__main__":
    main()
