"""
Streamlit page for RAG-based property analysis.
"""
import streamlit as st
import pandas as pd
from utils.vector_db_util import vector_db
from utils.db_util import db

st.set_page_config(page_title="RAG Analysis", page_icon="🧠")
st.title("🧠 RAG-based Property Analysis")

st.markdown("""
Find similar properties and analyze portfolio characteristics using vector embeddings and similarity search.
""")

# Check if database is initialized
if not db.check_initialized():
    st.warning("⚠️ Database not initialized. Please go to the Preprocess page and run the preprocessing step first.")
    st.stop()

@st.cache_resource
def initialize_vector_db():
    """Initialize and cache the vector database."""
    try:
        success = vector_db.initialize_vectors()
        if not success:
            st.error("Failed to initialize vector database")
            return None
        return vector_db
    except Exception as e:
        st.error(f"Error initializing vector database: {e}")
        return None

# Initialize vector database
with st.spinner("Loading vector database..."):
    rag_util = initialize_vector_db()

if rag_util is None:
    st.stop()

# Get marketed properties for selection
marketed_properties = rag_util.get_marketed_properties()

if len(marketed_properties) == 0:
    st.warning("No marketed properties found in the database.")
    st.stop()

st.markdown("### 🎯 Property Similarity Analysis")

# Property selection
col1, col2 = st.columns(2)

with col1:
    analysis_type = st.selectbox(
        "Choose analysis type:",
        ["Find Similar to Marketed Property", "Find Similar by Description", "Portfolio Homogeneity Analysis"]
    )

with col2:
    if analysis_type == "Find Similar to Marketed Property":
        selected_property_id = st.selectbox(
            "Select a marketed property:",
            marketed_properties['property_id'].tolist(),
            format_func=lambda x: f"ID {x}: {marketed_properties[marketed_properties['property_id']==x]['industrial_estate_name'].iloc[0]} - {marketed_properties[marketed_properties['property_id']==x]['unit_name'].iloc[0]}"
        )
    elif analysis_type == "Find Similar by Description":
        query_text = st.text_input(
            "Describe the property you're looking for:",
            value="Large industrial warehouse with good parking in the Midlands"
        )

# Analysis execution
if st.button("🔍 Run Analysis", type="primary"):
    with st.spinner("Analyzing properties using vector similarity..."):
        try:
            if analysis_type == "Find Similar to Marketed Property":
                result = rag_util.find_similar_properties(
                    target_property_id=selected_property_id,
                    top_k=10
                )
                
                if result['success']:
                    st.success(f"✅ Found {len(result['results'])} similar properties")
                    
                    # Display target property info
                    target_prop = marketed_properties[marketed_properties['property_id'] == selected_property_id].iloc[0]
                    st.markdown(f"**Target Property:** {target_prop['industrial_estate_name']} - {target_prop['unit_name']} ({target_prop['region']})")
                    
                    # Display results
                    results_df = pd.DataFrame(result['results'])
                    st.markdown("### 📊 Top 10 Most Similar Properties")
                    
                    # Format the display
                    display_cols = ['rank', 'industrial_estate_name', 'unit_name', 'region', 'size_sqm', 'build_year', 'similarity_score']
                    if all(col in results_df.columns for col in display_cols):
                        display_df = results_df[display_cols].copy()
                        display_df['similarity_score'] = display_df['similarity_score'].round(3)
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.dataframe(results_df, use_container_width=True)
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Similarity", f"{result['avg_similarity']:.3f}")
                    with col2:
                        st.metric("Max Similarity", f"{result['max_similarity']:.3f}")
                    with col3:
                        st.metric("Min Similarity", f"{result['min_similarity']:.3f}")
                        
                else:
                    st.error(f"Analysis failed: {result['error']}")
            
            elif analysis_type == "Find Similar by Description":
                if query_text.strip():
                    result = rag_util.find_similar_properties(
                        query_text=query_text,
                        top_k=8
                    )
                    
                    if result['success']:
                        st.success(f"✅ Found {len(result['results'])} properties matching your description")
                        
                        # Display results
                        results_df = pd.DataFrame(result['results'])
                        st.markdown("### 📊 Properties Matching Your Description")
                        
                        # Format the display
                        display_cols = ['rank', 'industrial_estate_name', 'unit_name', 'region', 'size_sqm', 'build_year', 'similarity_score']
                        if all(col in results_df.columns for col in display_cols):
                            display_df = results_df[display_cols].copy()
                            display_df['similarity_score'] = display_df['similarity_score'].round(3)
                            st.dataframe(display_df, use_container_width=True)
                        else:
                            st.dataframe(results_df, use_container_width=True)
                        
                        # Show statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Average Similarity", f"{result['avg_similarity']:.3f}")
                        with col2:
                            st.metric("Properties Searched", result['total_properties_searched'])
                    else:
                        st.error(f"Analysis failed: {result['error']}")
                else:
                    st.warning("Please enter a property description.")
            
            elif analysis_type == "Portfolio Homogeneity Analysis":
                result = rag_util.calculate_portfolio_homogeneity()
                
                if result['success']:
                    st.success("✅ Portfolio homogeneity analysis completed")
                    
                    st.markdown("### 📈 Portfolio Homogeneity Metrics")
                    
                    # Main metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Homogeneity", f"{result['overall_homogeneity']:.3f}")
                    with col2:
                        st.metric("Marketed vs Portfolio", f"{result['marketed_vs_portfolio']:.3f}")
                    with col3:
                        st.metric("Homogeneity Coefficient", f"{result['homogeneity_coefficient']:.3f}")
                    
                    # Additional metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Portfolio Internal Homogeneity", f"{result['portfolio_internal_homogeneity']:.3f}")
                    with col2:
                        st.metric("Marketed Internal Homogeneity", f"{result['marketed_internal_homogeneity']:.3f}")
                    
                    # Interpretation
                    st.markdown("### 📝 Interpretation")
                    if result['overall_homogeneity'] < 0.3:
                        st.info("🎯 **Diverse Portfolio**: Low homogeneity indicates a well-diversified property portfolio.")
                    elif result['overall_homogeneity'] < 0.6:
                        st.info("⚖️ **Balanced Portfolio**: Moderate homogeneity suggests balanced property characteristics.")
                    else:
                        st.info("🔄 **Homogeneous Portfolio**: High homogeneity indicates similar property characteristics.")
                    
                    # Distribution statistics
                    with st.expander("📊 Similarity Distribution Details"):
                        dist = result['similarity_distribution']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Minimum Similarity", f"{dist['min']:.3f}")
                            st.metric("25th Percentile", f"{dist['percentile_25']:.3f}")
                        with col2:
                            st.metric("Median Similarity", f"{dist['median']:.3f}")
                            st.metric("75th Percentile", f"{dist['percentile_75']:.3f}")
                        with col3:
                            st.metric("Maximum Similarity", f"{dist['max']:.3f}")
                            st.metric("Standard Deviation", f"{result['homogeneity_std']:.3f}")
                else:
                    st.error(f"Analysis failed: {result['error']}")
                    
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

# Information section
with st.expander("ℹ️ About RAG Analysis"):
    st.markdown("""
    **RAG (Retrieval-Augmented Generation) Analysis** uses vector embeddings to find similar properties based on:
    
    - **Property characteristics**: Size, build year, location, features
    - **Textual descriptions**: Estate names, regions, unit details
    - **Vector similarity**: Cosine similarity between property embeddings
    
    **Embedding Model**: TF-IDF Vectorization
    
    **Analysis Types**:
    - **Property Similarity**: Find properties similar to a specific marketed property
    - **Description Search**: Find properties matching a text description
    - **Portfolio Homogeneity**: Measure how similar properties are across the portfolio
    
    **Metrics**:
    - **Similarity Score**: Range from -1 (completely different) to 1 (identical)
    - **Homogeneity Coefficient**: Overall portfolio diversity measure
    - **Cross-similarity**: How marketed properties relate to the portfolio
    """)
