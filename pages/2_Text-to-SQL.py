'''
Streamlit page for Text-to-SQL functionality using simplified SQL chat.
'''
import streamlit as st
from utils.mock_sql_chat import MockSQLChat

st.set_page_config(page_title="Text-to-SQL", page_icon="🤖")

st.title("Text-to-SQL Querying with AI Agent")

st.write("Ask questions about the MLI property database in natural language. The AI agent will generate and execute SQL queries for you.")

# Initialize the SQL chat assistant
@st.cache_resource
def get_sql_assistant():
    return MockSQLChat()

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
    
    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Processing your question with AI agent..."):
                try:
                    result = sql_assistant.chat_with_database(user_question)
                    
                    if result["success"]:
                        st.write("### AI Agent Response")
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

