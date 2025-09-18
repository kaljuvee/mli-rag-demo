
import pandas as pd
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from .ai_util import get_embeddings_model

def create_vector_store(df: pd.DataFrame, columns_to_vectorize: list[str]) -> FAISS:
    """
    Creates a FAISS vector store from a DataFrame.
    """
    embeddings = get_embeddings_model()
    # Create a new column that concatenates the text from the specified columns
    df['vectorized_text'] = df[columns_to_vectorize].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    
    vector_store = FAISS.from_texts(df['vectorized_text'].tolist(), embeddings)
    return vector_store

def find_similar_properties(vector_store: FAISS, query_text: str, k: int = 10) -> list:
    """
    Finds the most similar properties in the vector store to a given query text.
    """
    similar_docs = vector_store.similarity_search(query_text, k=k)
    return similar_docs

