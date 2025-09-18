'''
Simple SQL query page without AI to test basic functionality.
'''
import streamlit as st
from utils import db_util

st.set_page_config(page_title="Simple Query", page_icon="🔍")

st.title("Simple SQL Query")

st.write("This page allows you to run basic SQL queries directly on the properties database.")

# Predefined queries
queries = {
    "Show all properties": "SELECT * FROM properties LIMIT 10",
    "Count total properties": "SELECT COUNT(*) as total_properties FROM properties",
    "Show marketed properties": "SELECT * FROM properties WHERE is_marketed = 1",
    "Largest properties": "SELECT industrial_estate_name, unit_name, size_sqm FROM properties ORDER BY size_sqm DESC LIMIT 10",
    "Properties by region": "SELECT region, COUNT(*) as count FROM properties GROUP BY region"
}

selected_query = st.selectbox("Select a query:", list(queries.keys()))

if st.button("Run Query"):
    try:
        sql_query = queries[selected_query]
        st.write("### SQL Query")
        st.code(sql_query, language="sql")
        
        results_df = db_util.query_db(sql_query)
        st.write("### Results")
        st.dataframe(results_df)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
