"""
SQL Chat utility using LangChain SQL Agent for MLI Demo.
Based on working implementation from ai-lending project.
"""
import os
import sqlite3
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class MLISQLChatAssistant:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = "/home/ubuntu/mli-rag-demo/db/mli.db"
        self.db_path = db_path
        
        # Initialize OpenAI client
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )
        
        # Initialize SQL database connection
        self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
        
        # Create SQL agent
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def get_database_schema(self) -> str:
        """Get a description of the database schema"""
        return self.db.get_table_info()

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
                "question": "Find the closest properties to the marketed property, after excluding any property in the portfolio that is more than 10 miles from a major city",
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

    def chat_with_database(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language question and return SQL results
        """
        try:
            # Add context about the database structure
            context_prompt = f"""
            You are a helpful AI assistant that can answer questions about a Multi-Let Industrial (MLI) property database.
            
            Database Schema:
            {self.get_database_schema()}
            
            Important Notes:
            - The database contains industrial property data with both current portfolio and marketed warehouses
            - Properties have physical characteristics: size_sqm, min_eaves_m, max_eaves_m, doors, yard_depth_m
            - Geospatial data: latitude, longitude for location analysis
            - Temporal data: build_year for age analysis
            - Marketing status: is_marketed (True for newly marketed properties, False for current portfolio)
            - EPC ratings indicate energy efficiency
            - All size measurements are in square meters and meters
            
            Please answer the following question: {question}
            
            If you need to write SQL, make sure to:
            1. Use proper table joins when needed
            2. Handle NULL values appropriately
            3. Use LIMIT clauses for large result sets
            4. Provide clear, business-friendly explanations
            5. For similarity analysis, consider size_sqm, region, build_year, and other relevant attributes
            """
            
            # Execute the query using the agent
            result = self.agent.run(context_prompt)
            
            return {
                "success": True,
                "answer": result,
                "question": question,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "answer": None,
                "question": question,
                "error": str(e)
            }

    def execute_direct_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute a direct SQL query (for advanced users)
        """
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
