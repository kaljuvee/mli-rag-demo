"""
Database utility for MLI property data.
Handles database operations using SQLite.
"""
import os
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text

class PropertyDatabase:
    def __init__(self, db_path=None):
        if db_path is None:
            # Default path: db/mli.db relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_dir = os.path.join(project_root, 'db')
            os.makedirs(db_dir, exist_ok=True)
            self.db_path = os.path.join(db_dir, 'mli.db')
        else:
            self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.conn = None
        self.engine = None
    
    def connect(self):
        """Connect to the database and return connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def get_engine(self):
        """Get SQLAlchemy engine for the database"""
        if not self.engine:
            self.engine = create_engine(f"sqlite:///{self.db_path}")
        return self.engine
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Properties table
        cursor.execute('''
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
        ''')
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_is_marketed ON properties(is_marketed)",
            "CREATE INDEX IF NOT EXISTS idx_region ON properties(region)",
            "CREATE INDEX IF NOT EXISTS idx_size ON properties(size_sqm)",
            "CREATE INDEX IF NOT EXISTS idx_build_year ON properties(build_year)",
            "CREATE INDEX IF NOT EXISTS idx_coordinates ON properties(latitude, longitude)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        self.close()
        return True
    
    def load_dataframe(self, df, table_name='properties', if_exists='replace'):
        """Load a pandas DataFrame into the database"""
        engine = self.get_engine()
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        return True
    
    def execute_query(self, query, params=None):
        """Execute a SQL query and return results as a list of dictionaries"""
        conn = self.connect()
        cursor = conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.commit()
        self.close()
        return results
    
    def query_to_df(self, query, params=None):
        """Execute a SQL query and return results as a pandas DataFrame"""
        engine = self.get_engine()
        
        if params:
            return pd.read_sql_query(query, engine, params=params)
        else:
            return pd.read_sql_query(query, engine)
    
    def get_table_info(self, table_name='properties'):
        """Get information about a table's schema"""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        self.close()
        
        return [dict(row) for row in columns]
    
    def get_row_count(self, table_name='properties', condition=None):
        """Get the number of rows in a table"""
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        if condition:
            query += f" WHERE {condition}"
        
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(query)
        count = cursor.fetchone()['count']
        self.close()
        
        return count
    
    def check_initialized(self):
        """Check if the database has been initialized with data"""
        try:
            count = self.get_row_count('properties')
            return count > 0
        except Exception:
            return False

# Create a global instance for convenience
db = PropertyDatabase()
