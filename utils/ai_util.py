'''
Utility for interacting with AI models using LangChain and OpenAI.
'''
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """Initializes and returns the ChatOpenAI model."""
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def get_embeddings_model():
    """Initializes and returns the OpenAI embeddings model."""
    return OpenAIEmbeddings()

def get_sql_from_prompt(prompt: str, schema: str) -> str:
    """
    Uses an LLM to convert a natural language prompt into a SQL query.
    """
    template = """
    Based on the table schema below, write a SQL query that would answer the user's question.

    ------------
    Schema:
    {schema}
    ------------
    Question:
    {question}
    ------------
    SQL Query:
    """
    prompt_template = ChatPromptTemplate.from_template(template)
    model = get_llm()

    chain = (
        prompt_template
        | model
        | StrOutputParser()
    )

    return chain.invoke({'schema': schema, 'question': prompt})

