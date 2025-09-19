"""
Utility for handling secrets in Streamlit Cloud.
This allows the application to run both locally and in Streamlit Cloud.
"""
import os
import streamlit as st

def get_streamlit_env():
    """
    Determine if running in Streamlit Cloud or locally.
    Returns 'cloud' or 'local'
    """
    # Check for Streamlit Cloud environment variables
    if os.environ.get('STREAMLIT_SHARING') or os.environ.get('IS_STREAMLIT_CLOUD'):
        return 'cloud'
    return 'local'

def get_data_path(filename):
    """
    Get the absolute path to a data file, handling both local and cloud environments.
    
    Args:
        filename: Name of the file in the data directory
        
    Returns:
        Absolute path to the file
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the project root
    project_root = os.path.dirname(current_dir)
    
    # Return the path to the data file
    return os.path.join(project_root, "data", filename)

def get_db_path():
    """
    Get the database path based on environment.
    
    Returns:
        Path to the SQLite database file
    """
    env = get_streamlit_env()
    
    if env == 'cloud':
        # In Streamlit Cloud, use /tmp directory which is writable
        return "/tmp/mli.db"
    else:
        # Locally, use the project directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        db_dir = os.path.join(project_root, "db")
        os.makedirs(db_dir, exist_ok=True)
        return os.path.join(db_dir, "mli.db")

def get_openai_api_key():
    """
    Get OpenAI API key from Streamlit secrets or environment variables.
    
    Returns:
        OpenAI API key or None if not found
    """
    # Try to get from Streamlit secrets
    try:
        return st.secrets["openai"]["api_key"]
    except:
        # Fall back to environment variable
        return os.environ.get("OPENAI_API_KEY")
