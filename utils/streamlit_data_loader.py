"""
Utility for loading data in Streamlit Cloud.
This handles the special case where data files need to be loaded from the repository.
"""
import os
import pandas as pd
import streamlit as st
from pathlib import Path

def get_streamlit_data_path():
    """
    Get the path to the data directory in Streamlit Cloud.
    In Streamlit Cloud, the repository is cloned to a specific location.
    """
    # Check if we're in Streamlit Cloud
    if os.environ.get('STREAMLIT_SHARING') or os.environ.get('IS_STREAMLIT_CLOUD'):
        # In Streamlit Cloud, the repository is cloned to this location
        return os.path.join(os.getcwd(), "data")
    else:
        # Locally, use the project directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        return os.path.join(project_root, "data")

@st.cache_data
def load_excel_data(filename):
    """
    Load Excel data with caching for Streamlit.
    
    Args:
        filename: Name of the Excel file in the data directory
        
    Returns:
        Pandas DataFrame with the Excel data
    """
    data_dir = get_streamlit_data_path()
    file_path = os.path.join(data_dir, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the Excel file
    return pd.read_excel(file_path)

def get_current_portfolio():
    """Get the current portfolio data."""
    return load_excel_data("CurrentPortfolio.xlsx")

def get_marketed_warehouses():
    """Get the marketed warehouses data."""
    return load_excel_data("MarketedWarehouses.xlsx")
