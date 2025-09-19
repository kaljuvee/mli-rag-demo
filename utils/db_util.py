'''
Utility for database operations using SQLAlchemy.
'''
import os
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path

# Use relative path for better compatibility across environments
def get_db_path():
    """Get the database path in a deployment-friendly way."""
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the project root
    project_root = os.path.dirname(current_dir)
    
    # Create db directory if it doesn't exist
    db_dir = os.path.join(project_root, "db")
    os.makedirs(db_dir, exist_ok=True)
    
    # Return the full path to the database file
    return os.path.join(db_dir, "mli.db")

# Get the database path
DB_FILE = get_db_path()

# Create the engine with the database path
ENGINE = create_engine(f"sqlite:///{DB_FILE}")

def get_engine():
    """Get the SQLAlchemy engine."""
    return ENGINE

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
