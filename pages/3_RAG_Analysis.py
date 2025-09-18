
import streamlit as st
import pandas as pd
from utils import db_util, rag_util

st.set_page_config(page_title="RAG Analysis", page_icon="🧠")
st.title("RAG-based Analysis")

@st.cache_resource
def get_vector_store():
    df = db_util.query_db("SELECT * FROM properties")
    # Define the columns to be used for vectorization
    columns_to_vectorize = ["industrial_estate_name", "unit_name", "region", "size_sqm", "build_year", "epc_rating"]
    vector_store = rag_util.create_vector_store(df, columns_to_vectorize)
    return vector_store, df

vector_store, df = get_vector_store()

marketed_properties = df[df["is_marketed"] == True]
selected_property = st.selectbox(
    "Select a marketed property to find similar properties:",
    marketed_properties["industrial_estate_name"].unique()
)

if st.button("Find Similar Properties"):
    if selected_property:
        with st.spinner("Finding similar properties..."):
            try:
                # Get the details of the selected marketed property
                property_details = marketed_properties[marketed_properties["industrial_estate_name"] == selected_property].iloc[0]
                
                # Create a query text from the property details
                query_text = f"{property_details['industrial_estate_name']} {property_details['unit_name']} in {property_details['region']}"

                # Find similar properties
                similar_docs = rag_util.find_similar_properties(vector_store, query_text, k=11) # k=11 to exclude the property itself

                st.write("### Top 10 Similar Properties")
                
                # Extract the text from the similar documents
                similar_properties_text = [doc.page_content for doc in similar_docs]
                
                # Find the corresponding rows in the original dataframe
                similar_properties_df = df[df["vectorized_text"].isin(similar_properties_text)]
                
                # Exclude the selected property from the results
                similar_properties_df = similar_properties_df[similar_properties_df["industrial_estate_name"] != selected_property]
                
                st.dataframe(similar_properties_df.head(10))

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please select a property.")

