'''
Utility for database operations using SQLAlchemy.
'''
import pandas as pd
from sqlalchemy import create_engine, text

DB_FILE = "/home/ubuntu/mli-rag-demo/db/mli.db"
ENGINE = create_engine(f"sqlite:///{DB_FILE}")

def get_engine():
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

