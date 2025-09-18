'''
FastAPI wrapper for the MLI demo application.
'''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import ai_util, db_util, rag_util
import pandas as pd

app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    type: str # 'sql' or 'rag'

@app.post("/chat")
def chat(request: ChatRequest):
    if request.type == "sql":
        try:
            # Get table schema
            with db_util.get_engine().connect() as connection:
                result = connection.execute(db_util.text("PRAGMA table_info(properties)"))
                schema = result.fetchall()
            schema_str = "\n".join([f"{row[1]} {row[2]}" for row in schema])

            # Generate SQL from prompt
            sql_query = ai_util.get_sql_from_prompt(request.query, schema_str)
            
            # Query the database
            results_df = db_util.query_db(sql_query)
            return results_df.to_dict(orient="records")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    elif request.type == "rag":
        try:
            df = db_util.query_db("SELECT * FROM properties")
            columns_to_vectorize = ["industrial_estate_name", "unit_name", "region", "size_sqm", "build_year", "epc_rating"]
            vector_store = rag_util.create_vector_store(df, columns_to_vectorize)

            similar_docs = rag_util.find_similar_properties(vector_store, request.query, k=10)
            
            similar_properties_text = [doc.page_content for doc in similar_docs]
            similar_properties_df = df[df["vectorized_text"].isin(similar_properties_text)]

            return similar_properties_df.to_dict(orient="records")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    else:
        raise HTTPException(status_code=400, detail="Invalid request type. Must be 'sql' or 'rag'.")

