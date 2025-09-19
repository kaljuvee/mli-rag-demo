"""
Mock SQL Chat utility for testing without OpenAI dependency.
"""
import os
import sqlite3
from typing import Dict, Any, List
import pandas as pd

from .db_util import db as default_db

class MockSQLChat:
    def __init__(self, db_path=None):
        if db_path is None:
            # Use the default database path from db_util
            self.db_path = default_db.db_path
        else:
            self.db_path = db_path

    def get_database_schema(self) -> str:
        """Get a description of the database schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get table info
            cursor.execute("PRAGMA table_info(properties)")
            columns = cursor.fetchall()
            
            schema_info = "Table: properties\nColumns:\n"
            for col in columns:
                schema_info += f"- {col[1]} ({col[2]})\n"
            
            # Get sample data
            cursor.execute("SELECT * FROM properties LIMIT 3")
            sample_data = cursor.fetchall()
            
            schema_info += f"\nSample data (first 3 rows):\n"
            for row in sample_data:
                schema_info += f"{row}\n"
            
            conn.close()
            return schema_info
            
        except Exception as e:
            return f"Error getting schema: {str(e)}"

    def generate_sql_query(self, question: str) -> str:
        """Generate SQL query from natural language question using predefined mappings"""
        
        # Predefined SQL queries for the 3 test cases
        query_mappings = {
            "find the 10 most similar properties": """
                SELECT p1.*, 
                       ABS(p1.size_sqm - m.avg_marketed_size) as size_diff,
                       ABS(p1.build_year - m.avg_marketed_year) as year_diff
                FROM properties p1,
                     (SELECT AVG(size_sqm) as avg_marketed_size, 
                             AVG(build_year) as avg_marketed_year
                      FROM properties WHERE is_marketed = 1) m
                WHERE p1.is_marketed = 0
                ORDER BY (ABS(p1.size_sqm - m.avg_marketed_size) + 
                         ABS(p1.build_year - m.avg_marketed_year)) ASC
                LIMIT 10
            """,
            
            "correlation score for the homogeneity": """
                SELECT 
                    'Physical Characteristics' as category,
                    CORR(size_sqm, min_eaves_m) as correlation_score,
                    COUNT(*) as sample_size
                FROM properties
                WHERE size_sqm IS NOT NULL AND min_eaves_m IS NOT NULL
                UNION ALL
                SELECT 
                    'Location Analysis' as category,
                    CORR(latitude, longitude) as correlation_score,
                    COUNT(*) as sample_size
                FROM properties
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                UNION ALL
                SELECT 
                    'Age Analysis' as category,
                    CORR(build_year, size_sqm) as correlation_score,
                    COUNT(*) as sample_size
                FROM properties
                WHERE build_year IS NOT NULL AND size_sqm IS NOT NULL
            """,
            
            "find the closest properties": """
                SELECT p1.*,
                       MIN(
                           6371 * acos(
                               cos(radians(m.latitude)) * 
                               cos(radians(p1.latitude)) * 
                               cos(radians(p1.longitude) - radians(m.longitude)) + 
                               sin(radians(m.latitude)) * 
                               sin(radians(p1.latitude))
                           )
                       ) as distance_km
                FROM properties p1
                CROSS JOIN (SELECT latitude, longitude FROM properties WHERE is_marketed = 1) m
                WHERE p1.is_marketed = 0
                  AND p1.latitude IS NOT NULL 
                  AND p1.longitude IS NOT NULL
                GROUP BY p1.property_id
                ORDER BY distance_km ASC
                LIMIT 10
            """,
            
            "show me all marketed properties": "SELECT * FROM properties WHERE is_marketed = 1",
            "count total properties": "SELECT COUNT(*) as total_properties FROM properties",
            "average size of properties": "SELECT AVG(size_sqm) as avg_size_sqm FROM properties WHERE size_sqm IS NOT NULL"
        }
        
        # Find matching query
        question_lower = question.lower()
        for key, sql in query_mappings.items():
            if key in question_lower:
                return sql.strip()
        
        # Default fallback
        return "SELECT * FROM properties LIMIT 10"

    def execute_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Handle CORR function which SQLite doesn't support natively
            if "CORR(" in sql_query:
                # For correlation queries, return mock correlation data
                mock_data = [
                    ("Physical Characteristics", 0.65, 1250),
                    ("Location Analysis", 0.23, 1250),
                    ("Age Analysis", 0.45, 1200)
                ]
                df = pd.DataFrame(mock_data, columns=["category", "correlation_score", "sample_size"])
            else:
                df = pd.read_sql_query(sql_query, conn)
            
            conn.close()
            
            return {
                "success": True,
                "data": df,
                "query": sql_query,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "query": sql_query,
                "error": str(e)
            }

    def chat_with_database(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language question and return SQL results
        """
        try:
            # Generate SQL query
            sql_query = self.generate_sql_query(question)
            
            # Execute the query
            result = self.execute_sql_query(sql_query)
            
            if result["success"]:
                # Format the answer
                df = result["data"]
                answer = f"Query executed successfully. Found {len(df)} results.\n\n"
                
                if len(df) > 0:
                    answer += "Results:\n"
                    answer += df.to_string(index=False)
                else:
                    answer += "No results found."
                
                return {
                    "success": True,
                    "answer": answer,
                    "sql_query": sql_query,
                    "data": df,
                    "question": question,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "answer": None,
                    "sql_query": sql_query,
                    "question": question,
                    "error": result["error"]
                }
                
        except Exception as e:
            return {
                "success": False,
                "answer": None,
                "sql_query": None,
                "question": question,
                "error": str(e)
            }

    def get_sample_queries(self) -> List[Dict[str, str]]:
        """Get sample queries users can try"""
        return [
            {
                "question": "Find the 10 most similar properties in the estate to the newly marketed property",
                "category": "Property Analysis"
            },
            {
                "question": "Provide a correlation score for the homogeneity of the marketed property(ies) with the rest of the estate",
                "category": "Portfolio Analysis"
            },
            {
                "question": "Find the closest properties to the marketed property",
                "category": "Geographic Analysis"
            }
        ]
