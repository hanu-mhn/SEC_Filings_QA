import warnings
warnings.filterwarnings("ignore", message="libmagic is unavailable")

import streamlit as st
from qa_agent import load_qa_chain, create_vector_store
import os

st.title("SEC Filings QA Agent")

# Create vector store if it doesn't exist
if not os.path.exists("vector_store"):
    st.warning("Vector store not found. Please run `python create_vector_store.py` to create it.")
    st.stop()

# Load the QA chain
qa_chain = load_qa_chain()

# Input for the user's question
question = st.text_input("Ask a question about the SEC filings:")

if question:
    with st.spinner("Searching for the answer..."):
        result = qa_chain({"query": question})
        st.write("### Answer")
        st.write(result["result"])

        st.write("### Source Documents")
        for doc in result["source_documents"]:
            st.write(f"- {doc.metadata['source']}")
