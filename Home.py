"""
MLI RAG Demo - Main Streamlit Application
"""
import streamlit as st
import pandas as pd
import os
from utils.db_util import db

# Page configuration
st.set_page_config(
    page_title="MLI RAG Demo",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .feature-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 10px;
    }
    .feature-desc {
        font-size: 1rem;
        color: #424242;
    }
    .stat-box {
        background-color: #e3f2fd;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #424242;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">MLI RAG Demo</div>', unsafe_allow_html=True)

# Introduction
st.markdown("""
This application demonstrates a production-ready implementation for analyzing Multi-Let Industrial (MLI) property data using both **Text-to-SQL** and **Retrieval-Augmented Generation (RAG)** techniques.

The demo showcases how AI can be leveraged to gain insights from property portfolio data, find similar properties, and analyze market trends using natural language queries.
""")

# Dashboard overview
st.markdown('<div class="sub-header">Dashboard Overview</div>', unsafe_allow_html=True)

# Check if database is initialized
db_initialized = db.check_initialized()
# Safely check if vector database is initialized by checking if embeddings directory exists
project_root = os.path.dirname(os.path.abspath(__file__))
embeddings_dir = os.path.join(project_root, 'embeddings')
vector_db_initialized = os.path.exists(embeddings_dir) and len(os.listdir(embeddings_dir)) > 0

# Stats row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{"1,255" if db_initialized else "0"}</div>
        <div class="stat-label">Total Properties</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{"5" if db_initialized else "0"}</div>
        <div class="stat-label">Marketed Properties</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{"✓" if db_initialized else "✗"}</div>
        <div class="stat-label">Database Status</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{"✓" if vector_db_initialized else "✗"}</div>
        <div class="stat-label">Vector DB Status</div>
    </div>
    """, unsafe_allow_html=True)

# System status message
if not db_initialized:
    st.warning("⚠️ Database not initialized. Please go to the **Preprocess** page to load the data.")
elif not vector_db_initialized:
    st.warning("⚠️ Vector database not initialized. Please go to the **Preprocess** page to build the vector index.")
else:
    st.success("✅ System fully initialized and ready for analysis!")

# Features section
st.markdown('<div class="sub-header">Key Features</div>', unsafe_allow_html=True)

# Feature boxes
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">🔍 Text-to-SQL Analysis</div>
        <div class="feature-desc">
            Ask questions about the property portfolio in natural language. The AI will convert your questions into SQL queries and execute them against the database.
            <br><br>
            <b>Example:</b> "Find the 10 most similar properties to the newly marketed property"
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">🧠 RAG-Based Similarity Search</div>
        <div class="feature-desc">
            Leverage vector embeddings to find similar properties based on multiple characteristics. The system uses TF-IDF vectorization and FAISS for efficient similarity search.
            <br><br>
            <b>Example:</b> Find properties similar to "Cherry Lane - Unit 10" based on size, location, and build year.
        </div>
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">📊 Portfolio Homogeneity Analysis</div>
        <div class="feature-desc">
            Analyze the consistency and diversity of your property portfolio. Calculate similarity scores between marketed and non-marketed properties to identify trends.
            <br><br>
            <b>Example:</b> "Provide a correlation score for the homogeneity of the marketed properties with the rest of the estate"
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">🌐 Geographic Analysis</div>
        <div class="feature-desc">
            Perform location-based analysis using property coordinates. Find properties within specific distances of major cities or other points of interest.
            <br><br>
            <b>Example:</b> "Find the closest properties to the marketed property, after excluding any property more than 10 miles from a major city"
        </div>
    </div>
    """, unsafe_allow_html=True)

# Getting started section
st.markdown('<div class="sub-header">Getting Started</div>', unsafe_allow_html=True)

st.markdown("""
Follow these steps to explore the MLI RAG Demo:

1. **Preprocess Data**: Start by visiting the **Preprocess** page to load the property data into the database and build the vector index.

2. **Text-to-SQL Queries**: Use the **Text-to-SQL** page to ask natural language questions about the property portfolio.

3. **RAG Analysis**: Explore the **RAG Analysis** page to find similar properties and analyze portfolio homogeneity.

4. **Simple Queries**: Use the **Simple Query** page for quick, predefined database queries.

Select a page from the sidebar to begin!
""")

# Sample data preview
if db_initialized:
    st.markdown('<div class="sub-header">Sample Data Preview</div>', unsafe_allow_html=True)
    
    try:
        # Get sample data
        sample_query = "SELECT * FROM properties LIMIT 5"
        sample_df = db.query_to_df(sample_query)
        
        # Display sample data
        st.dataframe(sample_df, use_container_width=True)
        
        # Show marketed properties
        st.markdown("### Marketed Properties")
        marketed_query = "SELECT * FROM properties WHERE is_marketed = 1"
        marketed_df = db.query_to_df(marketed_query)
        st.dataframe(marketed_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading sample data: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    MLI RAG Demo | Built with Streamlit, OpenAI, and FAISS | <a href="https://github.com/kaljuvee/mli-rag-demo" target="_blank">GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
