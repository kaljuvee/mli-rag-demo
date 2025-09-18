"""
Simplified SQL Chat utility for MLI Demo that works around OpenAI client issues.
"""
import os
import sqlite3
from typing import Dict, Any, List
import pandas as pd
from dotenv import load_dotenv
import openai

load_dotenv()

class SimpleSQLChat:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = "/home/ubuntu/mli-rag-demo/db/mli.db"
        self.db_path = db_path
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        """Generate SQL query from natural language question"""
        schema = self.get_database_schema()
        
        prompt = f"""
        You are a SQL expert. Given the following database schema and question, generate a SQL query.
        
        Database Schema:
        {schema}
        
        Question: {question}
        
        Important notes:
        - The table name is 'properties'
        - Use LIMIT clauses for large result sets
        - Handle NULL values appropriately
        - For similarity analysis, consider size_sqm, region, build_year, and other relevant attributes
        - is_marketed = 1 means newly marketed properties, is_marketed = 0 means current portfolio
        
        Return only the SQL query, no explanation:
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0
            )
            
            sql_query = response.choices[0].message.content.strip()
            # Clean up the SQL query
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            return sql_query
            
        except Exception as e:
            return f"Error generating SQL: {str(e)}"

    def execute_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            conn = sqlite3.connect(self.db_path)
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
            
            if sql_query.startswith("Error"):
                return {
                    "success": False,
                    "answer": None,
                    "sql_query": None,
                    "question": question,
                    "error": sql_query
                }
            
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
                "question": "Show me properties with similar size to the marketed warehouses",
                "category": "Size Analysis"
            },
            {
                "question": "Find properties in the same region as the marketed properties",
                "category": "Geographic Analysis"
            },
            {
                "question": "Show me the largest properties by size",
                "category": "Property Metrics"
            },
            {
                "question": "What is the average size of properties in each region?",
                "category": "Regional Analysis"
            },
            {
                "question": "Which properties have the highest EPC ratings?",
                "category": "Energy Efficiency"
            }
        ]
