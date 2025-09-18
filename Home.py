
import streamlit as st

st.set_page_config(
    page_title="MLI Demo",
    page_icon="🤖",
)

st.title("MLI Demo")

st.markdown("""
This application demonstrates a proof-of-concept for analyzing Multi-Let Industrial (MLI) data using both Text-to-SQL and Retrieval-Augmented Generation (RAG) techniques.

**👈 Select a demo from the sidebar** to see the different functionalities:

- **Preprocess**: Load, clean, and store the property data in a local SQLite database.
- **Text-to-SQL**: Ask questions about the property data in natural language.
- **RAG Analysis**: Find similar properties and analyze the portfolio using RAG.

""")

