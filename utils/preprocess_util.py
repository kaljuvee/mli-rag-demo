"""
Preprocessing utility for MLI property data.
Handles Excel file loading, data cleaning, and database creation.
"""

import os
import pandas as pd
import sqlite3
from typing import Tuple, Dict, Any, Optional
import numpy as np
from pathlib import Path

from .db_util import db as default_db


class PropertyPreprocessor:
    """Handles preprocessing of property data from Excel files."""
    
    def __init__(self, db_path: str = None):
        """Initialize preprocessor with database path."""
        # Use the path from db_util if none provided
        self.db_path = db_path if db_path else default_db.db_path
    
    def load_excel_files(self, 
                        current_portfolio_path: str,
                        marketed_warehouses_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load Excel files and return DataFrames.
        
        Args:
            current_portfolio_path: Path to CurrentPortfolio.xlsx
            marketed_warehouses_path: Path to MarketedWarehouses.xlsx
            
        Returns:
            Tuple of (current_portfolio_df, marketed_warehouses_df)
        """
        try:
            # Convert to absolute paths if needed
            current_portfolio_path = self._ensure_absolute_path(current_portfolio_path)
            marketed_warehouses_path = self._ensure_absolute_path(marketed_warehouses_path)
            
            print(f"Loading current portfolio from: {current_portfolio_path}")
            current_df = pd.read_excel(current_portfolio_path)
            print(f"✅ Loaded {len(current_df)} current properties")
            
            print(f"Loading marketed warehouses from: {marketed_warehouses_path}")
            marketed_df = pd.read_excel(marketed_warehouses_path)
            print(f"✅ Loaded {len(marketed_df)} marketed properties")
            
            return current_df, marketed_df
            
        except Exception as e:
            raise Exception(f"Error loading Excel files: {str(e)}")
    
    def _ensure_absolute_path(self, file_path: str) -> str:
        """Convert relative path to absolute path if needed."""
        if os.path.isabs(file_path):
            return file_path
        
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # Join with the relative path
        return os.path.join(project_root, file_path)
    
    def clean_and_standardize_data(self, 
                                  current_df: pd.DataFrame, 
                                  marketed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize property data.
        
        Args:
            current_df: Current portfolio DataFrame
            marketed_df: Marketed warehouses DataFrame
            
        Returns:
            Combined and cleaned DataFrame
        """
        print("🧹 Cleaning and standardizing data...")
        
        # Standardize column names
        def standardize_columns(df):
            df = df.copy()
            df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('#', '')
            return df
        
        current_clean = standardize_columns(current_df)
        marketed_clean = standardize_columns(marketed_df)
        
        # Add is_marketed flag
        current_clean['is_marketed'] = 0
        marketed_clean['is_marketed'] = 1
        
        # Combine datasets
        combined_df = pd.concat([current_clean, marketed_clean], ignore_index=True)
        
        # Handle missing values
        combined_df = self._handle_missing_values(combined_df)
        
        # Validate and clean data types
        combined_df = self._clean_data_types(combined_df)
        
        print(f"✅ Combined dataset: {len(combined_df)} total properties")
        print(f"   - Current properties: {len(current_clean)}")
        print(f"   - Marketed properties: {len(marketed_clean)}")
        
        return combined_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies."""
        df = df.copy()
        
        # Fill missing coordinates with 0 (will be handled in analysis)
        df['latitude'] = df['latitude'].fillna(0.0)
        df['longitude'] = df['longitude'].fillna(0.0)
        
        # Fill missing numeric values with 0
        numeric_cols = ['car_parking_spaces', 'size_sqm', 'build_year', 
                       'yard_depth_m', 'min._eaves_m', 'max._eaves_m', 'doors']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Fill missing categorical values
        df['epc_rating'] = df['epc_rating'].fillna('Not Available')
        df['region'] = df['region'].fillna('Unknown')
        
        return df
    
    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert data types."""
        df = df.copy()
        
        # Ensure numeric columns are properly typed
        numeric_conversions = {
            'property_id': 'float64',
            'latitude': 'float64',
            'longitude': 'float64',
            'car_parking_spaces': 'int64',
            'size_sqm': 'float64',
            'build_year': 'float64',
            'yard_depth_m': 'float64',
            'min._eaves_m': 'float64',
            'max._eaves_m': 'float64',
            'doors': 'int64',
            'is_marketed': 'int64'
        }
        
        for col, dtype in numeric_conversions.items():
            if col in df.columns:
                try:
                    if dtype.startswith('int'):
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                except Exception as e:
                    print(f"Warning: Could not convert {col} to {dtype}: {e}")
        
        # Ensure string columns are properly typed
        string_cols = ['industrial_estate_name', 'unit_name', 'region', 'epc_rating']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype('string').fillna('Unknown')
        
        return df
    
    def create_database_schema(self) -> bool:
        """Create database schema for properties."""
        try:
            # Create database connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Drop existing table if it exists
            cursor.execute("DROP TABLE IF EXISTS properties")
            
            # Create properties table
            create_table_sql = """
            CREATE TABLE properties (
                property_id REAL,
                industrial_estate_name TEXT,
                unit_name TEXT,
                region TEXT,
                latitude REAL,
                longitude REAL,
                car_parking_spaces INTEGER,
                size_sqm REAL,
                build_year REAL,
                yard_depth_m REAL,
                min_eaves_m REAL,
                max_eaves_m REAL,
                doors INTEGER,
                epc_rating TEXT,
                is_marketed INTEGER
            )
            """
            
            cursor.execute(create_table_sql)
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX idx_is_marketed ON properties(is_marketed)",
                "CREATE INDEX idx_region ON properties(region)",
                "CREATE INDEX idx_size ON properties(size_sqm)",
                "CREATE INDEX idx_build_year ON properties(build_year)",
                "CREATE INDEX idx_coordinates ON properties(latitude, longitude)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            conn.close()
            
            print(f"✅ Database schema created: {self.db_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating database schema: {e}")
            return False
    
    def load_data_to_database(self, df: pd.DataFrame) -> bool:
        """Load cleaned data into database."""
        try:
            # Rename columns to match database schema
            column_mapping = {
                'min._eaves_m': 'min_eaves_m',
                'max._eaves_m': 'max_eaves_m'
            }
            
            df_db = df.rename(columns=column_mapping)
            
            # Select only columns that exist in database schema
            db_columns = [
                'property_id', 'industrial_estate_name', 'unit_name', 'region',
                'latitude', 'longitude', 'car_parking_spaces', 'size_sqm',
                'build_year', 'yard_depth_m', 'min_eaves_m', 'max_eaves_m',
                'doors', 'epc_rating', 'is_marketed'
            ]
            
            # Only include columns that exist in the DataFrame
            available_columns = [col for col in db_columns if col in df_db.columns]
            df_final = df_db[available_columns]
            
            # Connect and load data
            conn = sqlite3.connect(self.db_path)
            
            # Clear existing data
            conn.execute("DELETE FROM properties")
            
            # Load new data
            df_final.to_sql('properties', conn, if_exists='append', index=False)
            
            # Verify data was loaded
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM properties")
            count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM properties WHERE is_marketed = 1")
            marketed_count = cursor.fetchone()[0]
            
            conn.close()
            
            print(f"✅ Data loaded to database:")
            print(f"   - Total properties: {count}")
            print(f"   - Marketed properties: {marketed_count}")
            print(f"   - Database file: {self.db_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data to database: {e}")
            return False
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the dataset."""
        try:
            stats = {
                "total_properties": len(df),
                "regions": df['region'].value_counts().to_dict(),
                "avg_size_sqm": float(df['size_sqm'].mean()) if 'size_sqm' in df.columns else 0,
                "avg_build_year": float(df['build_year'].mean()) if 'build_year' in df.columns else 0,
                "epc_ratings": df['epc_rating'].value_counts().to_dict() if 'epc_rating' in df.columns else {},
                "marketed_properties": int((df['is_marketed'] == 1).sum()),
                "properties_with_coordinates": int(((df['latitude'] != 0) & (df['longitude'] != 0)).sum()),
                "missing_data_summary": {
                    col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0
                }
            }
            return stats
        except Exception as e:
            print(f"Warning: Could not generate summary statistics: {e}")
            return {}


def run_preprocessing(current_portfolio_path: str = "data/CurrentPortfolio.xlsx",
                     marketed_warehouses_path: str = "data/MarketedWarehouses.xlsx",
                     db_path: str = None) -> Dict[str, Any]:
    """
    Convenience function to run preprocessing.
    
    Args:
        current_portfolio_path: Path to current portfolio Excel file
        marketed_warehouses_path: Path to marketed warehouses Excel file
        db_path: Path to SQLite database file (uses default_db.db_path if None)
        
    Returns:
        Dictionary with preprocessing results
    """
    preprocessor = PropertyPreprocessor(db_path=db_path)
    
    try:
        print("🚀 Starting MLI property data preprocessing...")
        
        # Step 1: Load Excel files
        current_df, marketed_df = preprocessor.load_excel_files(
            current_portfolio_path, marketed_warehouses_path
        )
        
        # Step 2: Clean and standardize data
        combined_df = preprocessor.clean_and_standardize_data(current_df, marketed_df)
        
        # Step 3: Create database schema
        if not preprocessor.create_database_schema():
            raise Exception("Failed to create database schema")
        
        # Step 4: Load data to database
        if not preprocessor.load_data_to_database(combined_df):
            raise Exception("Failed to load data to database")
        
        # Step 5: Generate summary statistics
        stats = preprocessor._generate_summary_stats(combined_df)
        
        print("✅ Preprocessing completed successfully!")
        
        return {
            "success": True,
            "total_properties": len(combined_df),
            "current_properties": len(current_df),
            "marketed_properties": len(marketed_df),
            "database_path": preprocessor.db_path,
            "statistics": stats,
            "sample_data": combined_df.head(3).to_dict('records'),
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Preprocessing failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        return {
            "success": False,
            "total_properties": 0,
            "current_properties": 0,
            "marketed_properties": 0,
            "database_path": preprocessor.db_path,
            "statistics": {},
            "sample_data": [],
            "error": error_msg
        }


if __name__ == "__main__":
    # Test preprocessing
    result = run_preprocessing()
    print("\nPreprocessing Result:")
    print(f"Success: {result['success']}")
    print(f"Total Properties: {result['total_properties']}")
    print(f"Database: {result['database_path']}")
    
    if result['error']:
        print(f"Error: {result['error']}")
