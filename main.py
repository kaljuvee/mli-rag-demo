"""
FastAPI application for the MLI RAG Demo.
Provides API endpoints for property analysis and SQL queries.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

# Create FastAPI app
app = FastAPI(
    title="MLI RAG Demo API",
    description="API for MLI Property Portfolio Analysis",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define request/response models
class ErrorResponse(BaseModel):
    success: bool = False
    error: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class PropertySearchRequest(BaseModel):
    query: str
    params: Dict[str, Any] = {}

# Direct database connection without dependencies
def get_db_connection():
    """Get SQLite database connection."""
    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
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

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Check if the API is running properly."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Get all properties endpoint
@app.get("/properties", tags=["properties"])
async def get_properties(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    marketed_only: bool = False
):
    """Retrieve a list of all properties with pagination."""
    try:
        # Build query
        query = "SELECT * FROM properties"
        if marketed_only:
            query += " WHERE is_marketed = 1"
        query += f" LIMIT {limit} OFFSET {offset}"
        
        # Execute query
        properties = query_to_dict(query)
        
        # Get total count
        count_query = "SELECT COUNT(*) as count FROM properties"
        if marketed_only:
            count_query += " WHERE is_marketed = 1"
        total = query_to_dict(count_query)[0]['count']
        
        return {
            "success": True,
            "count": len(properties),
            "total": total,
            "properties": properties,
            "error": None
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": str(e)}
        )

# Get property by ID endpoint
@app.get("/properties/{property_id}", tags=["properties"])
async def get_property_by_id(property_id: int):
    """Retrieve a specific property by its ID."""
    try:
        query = f"SELECT * FROM properties WHERE property_id = ?"
        properties = query_to_dict(query, (property_id,))
        
        if not properties:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "error": f"Property with ID {property_id} not found"}
            )
        
        return {
            "success": True,
            "property": properties[0],
            "error": None
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": str(e)}
        )

# Property search endpoint
@app.post("/properties/search", tags=["properties"])
async def search_properties(request: PropertySearchRequest):
    """Search properties using SQL-like query string."""
    try:
        # Execute query
        properties = query_to_dict(request.query, request.params)
        
        return {
            "success": True,
            "count": len(properties),
            "properties": properties,
            "query": request.query,
            "error": None
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST if "syntax error" in str(e).lower() else status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": f"Query error: {str(e)}"}
        )

# Get marketed properties endpoint
@app.get("/properties/marketed", tags=["properties"])
async def get_marketed_properties():
    """Retrieve all marketed properties."""
    try:
        query = "SELECT * FROM properties WHERE is_marketed = 1"
        properties = query_to_dict(query)
        
        return {
            "success": True,
            "count": len(properties),
            "properties": properties,
            "error": None
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": str(e)}
        )

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    # Try to load from swagger.json if it exists
    swagger_path = os.path.join(os.path.dirname(__file__), "api", "swagger.json")
    if os.path.exists(swagger_path):
        try:
            with open(swagger_path, "r") as f:
                app.openapi_schema = json.load(f)
                return app.openapi_schema
        except:
            pass
    
    # Fall back to auto-generated schema
    openapi_schema = get_openapi(
        title="MLI RAG Demo API",
        version="1.0.0",
        description="API for MLI Property Portfolio Analysis",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
