"""
FastAPI wrapper for the MLI demo application.
Simplified version with minimal dependencies.
"""
import os
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    type: str  # 'sql' or 'rag'

# Direct database connection without dependencies
def get_db_connection():
    """Get SQLite database connection."""
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_dir = os.path.join(project_root, "db")
    db_path = os.path.join(db_dir, "mli.db")
    
    # Create directory if it doesn't exist
    os.makedirs(db_dir, exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def query_to_dict(query, params=None):
    """Execute SQL query and return results as list of dictionaries."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        # Get column names
        columns = [col[0] for col in cursor.description]
        
        # Convert rows to dictionaries
        results = []
        for row in cursor.fetchall():
            results.append({columns[i]: row[i] for i in range(len(columns))})
        
        return results
    finally:
        conn.close()

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/chat")
def chat(request: ChatRequest):
    """Chat endpoint for SQL and RAG queries."""
    if request.type == "sql":
        try:
            # For SQL queries, just execute the query directly
            # This is a simplified version that assumes the query is safe
            results = query_to_dict(request.query)
            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    elif request.type == "rag":
        try:
            # For RAG queries, we'll do a simple text search instead of vector search
            # This is a simplified version that doesn't require external dependencies
            search_terms = request.query.lower().split()
            
            # Get all properties
            all_properties = query_to_dict("SELECT * FROM properties")
            
            # Simple text search function
            def simple_match_score(property_dict, search_terms):
                text = f"{property_dict.get('industrial_estate_name', '')} {property_dict.get('unit_name', '')} {property_dict.get('region', '')}"
                text = text.lower()
                score = sum(1 for term in search_terms if term in text)
                return score
            
            # Score and sort properties
            scored_properties = [(prop, simple_match_score(prop, search_terms)) for prop in all_properties]
            scored_properties.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 10 matches
            return [prop for prop, score in scored_properties[:10] if score > 0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    else:
        raise HTTPException(status_code=400, detail="Invalid request type. Must be 'sql' or 'rag'.")

# Add a root endpoint that redirects to the main API
@app.get("/")
def root():
    """Redirect to the main API."""
    return {"message": "Welcome to MLI Demo API. Please use /chat endpoint for queries."}
