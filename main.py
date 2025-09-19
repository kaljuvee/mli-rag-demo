"""
FastAPI application for the MLI RAG Demo.
Provides API endpoints for property analysis, Text-to-SQL, and RAG functionality.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

from utils import db_util, vector_db_util

# Initialize databases
db = db_util.PropertyDatabase()
vector_db = vector_db_util.PropertyVectorDB()

# Create FastAPI app
app = FastAPI(
    title="MLI RAG Demo API",
    description="API for MLI Property Portfolio Analysis with RAG and Text-to-SQL capabilities",
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

class TextToSqlRequest(BaseModel):
    question: str
    include_sql: bool = True

class TextToSqlResponse(BaseModel):
    success: bool
    question: str
    sql: Optional[str] = None
    results: List[Dict[str, Any]]
    execution_time: float
    error: Optional[str] = None

class SimilarPropertiesRequest(BaseModel):
    property_id: Optional[int] = None
    query_text: Optional[str] = None
    top_k: int = 10

class RagQueryRequest(BaseModel):
    question: str
    include_context: bool = True

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
        properties_df = db.query_to_df(query)
        
        # Get total count
        count_query = "SELECT COUNT(*) as count FROM properties"
        if marketed_only:
            count_query += " WHERE is_marketed = 1"
        total_df = db.query_to_df(count_query)
        total = int(total_df['count'].iloc[0]) if not total_df.empty else 0
        
        return {
            "success": True,
            "count": len(properties_df),
            "total": total,
            "properties": properties_df.to_dict(orient="records"),
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
        query = f"SELECT * FROM properties WHERE property_id = {property_id}"
        property_df = db.query_to_df(query)
        
        if property_df.empty:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "error": f"Property with ID {property_id} not found"}
            )
        
        return {
            "success": True,
            "property": property_df.iloc[0].to_dict(),
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
        properties_df = db.query_to_df(request.query, params=request.params)
        
        return {
            "success": True,
            "count": len(properties_df),
            "properties": properties_df.to_dict(orient="records"),
            "query": request.query,
            "error": None
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST if "syntax error" in str(e).lower() else status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": f"Query error: {str(e)}"}
        )

# Text-to-SQL endpoint
@app.post("/sql/query", response_model=TextToSqlResponse, tags=["sql"])
async def text_to_sql_query(request: TextToSqlRequest):
    """Convert natural language question to SQL and execute it."""
    try:
        from utils.mock_sql_chat import MockSQLChat
        import time
        
        start_time = time.time()
        
        # Initialize SQL chat
        sql_chat = MockSQLChat()
        
        # Generate SQL from question
        result = sql_chat.generate_sql_response(request.question)
        
        execution_time = time.time() - start_time
        
        return {
            "success": True,
            "question": request.question,
            "sql": result.get("sql") if request.include_sql else None,
            "results": result.get("results", []),
            "execution_time": execution_time,
            "error": None
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "question": request.question,
                "sql": None,
                "results": [],
                "execution_time": 0,
                "error": str(e)
            }
        )

# Find similar properties endpoint
@app.post("/rag/similar", tags=["rag"])
async def find_similar_properties(request: SimilarPropertiesRequest):
    """Find properties similar to a target property using vector similarity."""
    try:
        # Initialize vector database if needed
        if not vector_db.initialize_vectors():
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "error": "Failed to initialize vector database"}
            )
        
        # Find similar properties
        result = vector_db.find_similar_properties(
            target_property_id=request.property_id,
            query_text=request.query_text,
            top_k=request.top_k
        )
        
        if not result["success"]:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND if "not found" in result["error"].lower() else status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "error": result["error"]}
            )
        
        return result
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": str(e)}
        )

# Portfolio homogeneity endpoint
@app.get("/rag/homogeneity", tags=["rag"])
async def portfolio_homogeneity():
    """Calculate portfolio homogeneity using vector embeddings."""
    try:
        # Initialize vector database if needed
        if not vector_db.initialize_vectors():
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "error": "Failed to initialize vector database"}
            )
        
        # Calculate homogeneity
        result = vector_db.calculate_portfolio_homogeneity()
        
        if not result["success"]:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "error": result["error"]}
            )
        
        return result
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": str(e)}
        )

# RAG query endpoint
@app.post("/rag/query", tags=["rag"])
async def rag_query(request: RagQueryRequest):
    """Query the property database using RAG approach."""
    try:
        # This is a placeholder for actual RAG implementation
        # In a real implementation, this would use a language model with the vector database
        
        # Initialize vector database if needed
        if not vector_db.initialize_vectors():
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "error": "Failed to initialize vector database"}
            )
        
        # For now, return a mock response based on the question
        if "homogeneity" in request.question.lower():
            homogeneity = vector_db.calculate_portfolio_homogeneity()
            
            answer = f"The portfolio has a homogeneity score of {homogeneity['overall_homogeneity']:.2f}, "
            if homogeneity['overall_homogeneity'] < 0.3:
                answer += "indicating a diverse property portfolio with varied characteristics."
            elif homogeneity['overall_homogeneity'] < 0.6:
                answer += "suggesting a balanced mix of property characteristics across the estate."
            else:
                answer += "showing a highly homogeneous property portfolio with similar characteristics."
                
            context = []
            
        elif "similar" in request.question.lower() and "marketed" in request.question.lower():
            # Get a marketed property
            marketed_props = vector_db.get_marketed_properties()
            if len(marketed_props) > 0:
                prop_id = marketed_props.iloc[0]['property_id']
                similar = vector_db.find_similar_properties(target_property_id=prop_id, top_k=5)
                
                answer = f"The most similar properties to the marketed property '{marketed_props.iloc[0]['industrial_estate_name']}' "
                answer += f"are located in regions like {', '.join(set([p['region'] for p in similar['results'][:3]]))}. "
                answer += f"These properties have similarity scores ranging from {similar['min_similarity']:.2f} to {similar['max_similarity']:.2f}."
                
                context = similar['results'][:3] if request.include_context else []
            else:
                answer = "No marketed properties found to compare with."
                context = []
        else:
            answer = "I can analyze the property portfolio for you. You can ask about portfolio homogeneity, "
            answer += "find similar properties to marketed ones, or explore property characteristics across regions."
            context = []
        
        return {
            "success": True,
            "question": request.question,
            "answer": answer,
            "context": context if request.include_context else [],
            "confidence": 0.85,
            "execution_time": 0.75,
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
        description="API for MLI Property Portfolio Analysis with RAG and Text-to-SQL capabilities",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
