'''
Streamlit page for Text-to-SQL functionality.
'''
import streamlit as st
from utils import ai_util, db_util

st.set_page_config(page_title="Text-to-SQL", page_icon="🤖")

st.title("Text-to-SQL Querying")

# Get table schema
with db_util.get_engine().connect() as connection:
    result = connection.execute(db_util.text("PRAGMA table_info(properties)"))
    schema = result.fetchall()

schema_str = "\n".join([f"{row[1]} {row[2]}" for row in schema])

with st.expander("View Table Schema"):
    st.code(schema_str, language="sql")

user_question = st.text_input("Ask a question about the properties:", "Find the 10 largest properties by size")

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Generating SQL query and fetching results..."):
            try:
                # Generate SQL from prompt
                sql_query = ai_util.get_sql_from_prompt(user_question, schema_str)
                st.write("### Generated SQL Query")
                st.code(sql_query, language="sql")

                # Query the database
                results_df = db_util.query_db(sql_query)

                st.write("### Results")
                st.dataframe(results_df)

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")

