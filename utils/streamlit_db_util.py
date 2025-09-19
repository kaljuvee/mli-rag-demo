'''
Streamlit-specific database utility for cloud deployment.
This utility handles database operations in a way that's compatible with Streamlit Cloud.
'''
import os
import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st

from .streamlit_secrets import get_db_path

# Get the database path
DB_FILE = get_db_path()

# Create the engine with the database path
@st.cache_resource
def get_engine():
    """Get the SQLAlchemy engine with caching for Streamlit."""
    engine = create_engine(f"sqlite:///{DB_FILE}")
    # Create tables if they don't exist
    with engine.connect() as connection:
        connection.execute(text("""
        CREATE TABLE IF NOT EXISTS properties (
            property_id INTEGER PRIMARY KEY,
            industrial_estate_name TEXT,
            unit_name TEXT,
            region TEXT,
            latitude REAL,
            longitude REAL,
            car_parking_spaces INTEGER,
            size_sqm REAL,
            build_year INTEGER,
            yard_depth_m REAL,
            min_eaves_m REAL,
            max_eaves_m REAL,
            doors INTEGER,
            epc_rating TEXT,
            is_marketed BOOLEAN
        )
        """))
    return engine

ENGINE = get_engine()

def create_tables():
    """Creates the properties table in the database."""
    with ENGINE.connect() as connection:
        connection.execute(text("""
        CREATE TABLE IF NOT EXISTS properties (
            property_id INTEGER PRIMARY KEY,
            industrial_estate_name TEXT,
            unit_name TEXT,
            region TEXT,
            latitude REAL,
            longitude REAL,
            car_parking_spaces INTEGER,
            size_sqm REAL,
            build_year INTEGER,
            yard_depth_m REAL,
            min_eaves_m REAL,
            max_eaves_m REAL,
            doors INTEGER,
            epc_rating TEXT,
            is_marketed BOOLEAN
        )
        """))

def load_df_to_db(df: pd.DataFrame, table_name: str):
    """Loads a DataFrame into the specified database table."""
    df.to_sql(table_name, con=ENGINE, if_exists='replace', index=False)

def query_db(query: str) -> pd.DataFrame:
    """Queries the database and returns a DataFrame."""
    with ENGINE.connect() as connection:
        return pd.read_sql(query, connection)

def check_db_initialized() -> bool:
    """Check if the database has been initialized with data."""
    try:
        with ENGINE.connect() as connection:
            result = connection.execute(text("SELECT COUNT(*) FROM properties"))
            count = result.scalar()
            return count > 0
    except Exception:
        return False
