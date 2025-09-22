"""
Streamlit page for Text-to-SQL functionality using simplified SQL chat.
"""
import streamlit as st
import pandas as pd
import json
import time
from utils.sql_chat import MLISQLChatAssistant
from utils.db_util import db

# Page configuration
st.set_page_config(
    page_title="Text-to-SQL",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .result-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .sql-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        font-family: monospace;
    }
    .success-box {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }
    .metrics-container {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-around;
    }
    .metric-item {
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #424242;
    }
    .json-viewer {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 10px;
        max-height: 300px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">Text-to-SQL Querying with AI Agent</div>', unsafe_allow_html=True)

st.write("Ask questions about the MLI property database in natural language. The AI agent will generate and execute SQL queries for you.")

# Check if database is initialized
if not db.check_initialized():
    st.warning("⚠️ Database not initialized. Please go to the Preprocess page and run the preprocessing step first.")
    st.stop()

# Initialize the SQL chat assistant
@st.cache_resource
def get_sql_assistant():
    return MLISQLChatAssistant(db_path=db.db_path)

try:
    sql_assistant = get_sql_assistant()
    
    # Show database schema
    with st.expander("View Database Schema"):
        schema = sql_assistant.get_database_schema()
        st.code(schema, language="sql")
    
    # Show sample queries
    with st.expander("Sample Questions You Can Ask"):
        sample_queries = sql_assistant.get_sample_queries()
        for query in sample_queries:
            st.write(f"**{query['category']}**: {query['question']}")
    
    # User input
    user_question = st.text_input(
        "Ask a question about the properties:", 
        "Find the 10 most similar properties in the estate to the newly marketed property"
    )
    
    if st.button("Get Answer", type="primary"):
        if user_question:
            with st.spinner("Processing your question with AI agent..."):
                try:
                    start_time = time.time()
                    result = sql_assistant.chat_with_database(user_question)
                    execution_time = time.time() - start_time
                    
                    if result["success"]:
                        # Success message
                        st.markdown('<div class="success-box">Assistant response generated.</div>', unsafe_allow_html=True)
                        
                        # Display metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Execution Time", f"{execution_time:.2f}s")
                        with col2:
                            st.metric("Model", "gpt-4o-mini")
                        
                        # Display assistant answer
                        st.markdown("### AI Agent Response")
                        st.write(result["answer"])
                    else:
                        st.error(f"An error occurred: {result['error']}")
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question.")
            
except Exception as e:
    st.error(f"Failed to initialize SQL assistant: {e}")
    st.write("Please ensure the database has been created by running the Preprocess step first.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    Text-to-SQL functionality powered by LangChain and SQLite | <a href="https://github.com/kaljuvee/mli-rag-demo" target="_blank">GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
