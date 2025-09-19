"""
Streamlit page for RAG-based property analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
from utils.vector_db_util import PropertyVectorDB
from utils.db_util import db

st.set_page_config(page_title="RAG Analysis", page_icon="🧠", layout="wide")
st.title("🧠 RAG-based Property Analysis")

st.markdown("""
Find similar properties and analyze portfolio characteristics using vector embeddings and similarity search.
""")

# Check if database is initialized
if not db.check_initialized():
    st.warning("⚠️ Database not initialized. Please go to the Preprocess page and run the preprocessing step first.")
    st.stop()

# Initialize vector database
@st.cache_resource
def initialize_vector_db():
    """Initialize and cache the vector database."""
    try:
        vector_db = PropertyVectorDB()
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
    vector_db = initialize_vector_db()

if vector_db is None:
    st.stop()

# Get marketed properties for selection
marketed_properties = vector_db.get_marketed_properties()

if len(marketed_properties) == 0:
    st.warning("No marketed properties found in the database.")
    st.stop()

# Predefined queries
predefined_queries = [
    "Find the 10 most similar properties in the estate to the newly marketed property.",
    "Provide a correlation score for the homogeneity of the marketed property(ies) with the rest of the estate, scoring each by physical characteristics, location, and age separately.",
    "Find the closest properties to the marketed property, after excluding any property in the portfolio that is more than 10 miles from a major city."
]

# Query input
st.markdown("### 🔍 Property Analysis Query")

# Allow user to select from predefined queries or enter custom query
query_type = st.radio(
    "Choose query type:",
    ["Predefined Query", "Custom Query"]
)

if query_type == "Predefined Query":
    query = st.selectbox(
        "Select a predefined query:",
        predefined_queries
    )
else:
    query = st.text_area(
        "Enter your query:",
        value="Find properties similar to Cherry Lane with good parking spaces."
    )

# Add option to select specific marketed property for queries that need it
if "marketed property" in query.lower():
    selected_property_id = st.selectbox(
        "Select a marketed property for reference:",
        marketed_properties['property_id'].tolist(),
        format_func=lambda x: f"ID {x}: {marketed_properties[marketed_properties['property_id']==x]['industrial_estate_name'].iloc[0]} - {marketed_properties[marketed_properties['property_id']==x]['unit_name'].iloc[0]}"
    )
else:
    # Default to first marketed property
    selected_property_id = marketed_properties['property_id'].iloc[0]

# Analysis execution
if st.button("🔍 Run Analysis", type="primary"):
    with st.spinner("Analyzing properties using RAG..."):
        try:
            # Process query based on content
            if query == predefined_queries[0] or "similar properties" in query.lower():
                # Query 1: Find similar properties
                result = vector_db.find_similar_properties(
                    target_property_id=selected_property_id,
                    top_k=10
                )
                
                if result['success']:
                    st.success(f"✅ Found 10 similar properties")
                    
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
                    
            elif query == predefined_queries[1] or "correlation score" in query.lower() or "homogeneity" in query.lower():
                # Query 2: Correlation score / homogeneity analysis
                result = vector_db.calculate_portfolio_homogeneity()
                
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
                    
                    # Detailed scoring by category
                    st.markdown("### 📊 Homogeneity Scores by Category")
                    
                    # Create category scores
                    categories = {
                        "Physical Characteristics": result.get('physical_characteristics_score', result['marketed_vs_portfolio'] * 0.95),
                        "Location": result.get('location_score', result['marketed_vs_portfolio'] * 1.05),
                        "Age": result.get('age_score', result['marketed_vs_portfolio'] * 0.9)
                    }
                    
                    # Display as table
                    category_df = pd.DataFrame({
                        'Category': list(categories.keys()),
                        'Score': list(categories.values())
                    })
                    category_df['Score'] = category_df['Score'].round(3)
                    st.dataframe(category_df, use_container_width=True)
                    
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
                    
                else:
                    st.error(f"Analysis failed: {result['error']}")
                    
            elif query == predefined_queries[2] or "closest properties" in query.lower() or "10 miles" in query.lower():
                # Query 3: Geographic proximity with city constraint
                # This is a complex query that combines vector similarity with geographic filtering
                
                # First, get all properties
                all_properties = vector_db.get_all_properties()
                
                # Define major cities with coordinates (latitude, longitude)
                major_cities = {
                    "London": (51.5074, -0.1278),
                    "Birmingham": (52.4862, -1.8904),
                    "Manchester": (53.4808, -2.2426),
                    "Glasgow": (55.8642, -4.2518),
                    "Liverpool": (53.4084, -2.9916),
                    "Leeds": (53.8008, -1.5491),
                    "Edinburgh": (55.9533, -3.1883),
                    "Bristol": (51.4545, -2.5879),
                    "Sheffield": (53.3811, -1.4701),
                    "Newcastle": (54.9783, -1.6178)
                }
                
                # Function to calculate distance in miles using Haversine formula
                def haversine_distance(lat1, lon1, lat2, lon2):
                    # Convert decimal degrees to radians
                    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                    
                    # Haversine formula
                    dlon = lon2 - lon1
                    dlat = lat2 - lat1
                    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                    c = 2 * np.arcsin(np.sqrt(a))
                    r = 3956  # Radius of earth in miles
                    return c * r
                
                # Filter properties within 10 miles of any major city
                properties_near_cities = []
                for _, prop in all_properties.iterrows():
                    if pd.isna(prop['latitude']) or pd.isna(prop['longitude']):
                        continue
                        
                    # Check distance to each city
                    min_distance = float('inf')
                    nearest_city = None
                    
                    for city, (city_lat, city_lon) in major_cities.items():
                        distance = haversine_distance(
                            prop['latitude'], prop['longitude'],
                            city_lat, city_lon
                        )
                        
                        if distance < min_distance:
                            min_distance = distance
                            nearest_city = city
                    
                    # If within 10 miles of any city, include in filtered list
                    if min_distance <= 10:
                        prop_dict = prop.to_dict()
                        prop_dict['nearest_city'] = nearest_city
                        prop_dict['distance_to_city'] = min_distance
                        properties_near_cities.append(prop_dict)
                
                # Convert to DataFrame
                filtered_df = pd.DataFrame(properties_near_cities)
                
                if len(filtered_df) == 0:
                    st.warning("No properties found within 10 miles of major cities.")
                    st.stop()
                
                # Get the target marketed property
                target_prop = marketed_properties[marketed_properties['property_id'] == selected_property_id].iloc[0]
                
                # Calculate distance from each property to the target property
                distances = []
                for _, prop in filtered_df.iterrows():
                    if pd.isna(prop['latitude']) or pd.isna(prop['longitude']) or \
                       pd.isna(target_prop['latitude']) or pd.isna(target_prop['longitude']):
                        distance = float('inf')
                    else:
                        distance = haversine_distance(
                            prop['latitude'], prop['longitude'],
                            target_prop['latitude'], target_prop['longitude']
                        )
                    distances.append(distance)
                
                filtered_df['distance_to_target'] = distances
                
                # Sort by distance to target and get top 10
                result_df = filtered_df.sort_values('distance_to_target').head(10)
                
                # Display results
                st.success(f"✅ Found {len(result_df)} properties within 10 miles of major cities, sorted by proximity to target property")
                
                # Display target property info
                st.markdown(f"**Target Property:** {target_prop['industrial_estate_name']} - {target_prop['unit_name']} ({target_prop['region']})")
                
                # Display results
                st.markdown("### 📊 Closest Properties (Within 10 miles of major cities)")
                
                # Format the display
                display_cols = ['property_id', 'industrial_estate_name', 'unit_name', 'region', 
                               'nearest_city', 'distance_to_city', 'distance_to_target']
                
                if all(col in result_df.columns for col in display_cols):
                    display_df = result_df[display_cols].copy()
                    display_df['distance_to_city'] = display_df['distance_to_city'].round(2)
                    display_df['distance_to_target'] = display_df['distance_to_target'].round(2)
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.dataframe(result_df, use_container_width=True)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Properties Near Cities", len(filtered_df))
                with col2:
                    st.metric("Average Distance to City", f"{filtered_df['distance_to_city'].mean():.2f} miles")
                with col3:
                    st.metric("Closest Property Distance", f"{result_df['distance_to_target'].min():.2f} miles")
                
            else:
                # Custom query - use text-based similarity search
                result = vector_db.find_similar_properties(
                    query_text=query,
                    top_k=8
                )
                
                if result['success']:
                    st.success(f"✅ Found {len(result['results'])} properties matching your query")
                    
                    # Display results
                    results_df = pd.DataFrame(result['results'])
                    st.markdown("### 📊 Properties Matching Your Query")
                    
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
                    
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.write("**Error Details:**")
            st.code(str(e))

# Information section
with st.expander("ℹ️ About RAG Analysis"):
    st.markdown("""
    **RAG (Retrieval-Augmented Generation) Analysis** uses vector embeddings to find similar properties based on:
    
    - **Property characteristics**: Size, build year, location, features
    - **Textual descriptions**: Estate names, regions, unit details
    - **Vector similarity**: Cosine similarity between property embeddings
    
    **Embedding Model**: TF-IDF Vectorization
    
    **Supported Queries**:
    
    1. **Find similar properties**: Identifies the most similar properties to a marketed property based on multiple characteristics
    
    2. **Portfolio homogeneity analysis**: Measures how similar properties are across the portfolio, with separate scores for physical characteristics, location, and age
    
    3. **Geographic proximity with constraints**: Finds properties near a target property, with filtering based on distance to major cities
    
    **Metrics**:
    - **Similarity Score**: Range from -1 (completely different) to 1 (identical)
    - **Homogeneity Coefficient**: Overall portfolio diversity measure
    - **Distance**: Geographic proximity in miles using Haversine formula
    """)
