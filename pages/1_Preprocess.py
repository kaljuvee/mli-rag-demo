"""
Streamlit page for preprocessing the MLI property data.
"""
import streamlit as st
import pandas as pd
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess_util import PropertyPreprocessor
from utils.streamlit_db_util import DB_FILE, check_db_initialized
from utils.streamlit_data_loader import get_current_portfolio, get_marketed_warehouses

st.set_page_config(page_title="Preprocess Data", page_icon="⚙️")

st.title("⚙️ Preprocess and Load Data")

st.write("""
This page loads the Excel files, cleans the data, and stores it in a local SQLite database.

**Data Sources:**
- CurrentPortfolio.xlsx: 1,250 current properties
- MarketedWarehouses.xlsx: 5 marketed properties

**Processing Steps:**
1. Load and validate Excel files
2. Clean and standardize data formats
3. Handle missing values with appropriate strategies
4. Create optimized SQLite database schema
5. Load data with proper indexing for performance
""")

# Initialize session state for preprocessing status
if 'preprocess_done' not in st.session_state:
    st.session_state.preprocess_done = check_db_initialized()

# Check if data is already loaded
if st.session_state.preprocess_done:
    st.success("✅ Data is already loaded in the database.")
    
    # Option to reload data
    if st.button("Reload Data"):
        st.session_state.preprocess_done = False

if st.button("Run Preprocessing", type="primary") or st.session_state.get('preprocess_done') == False:
    try:
        with st.spinner("Loading and processing data..."):
            # Load data directly using the streamlit_data_loader
            current_df = get_current_portfolio()
            marketed_df = get_marketed_warehouses()
            
            st.info(f"Loaded {len(current_df)} current properties and {len(marketed_df)} marketed properties")
            
            # Initialize preprocessor with the Streamlit-specific database path
            preprocessor = PropertyPreprocessor(db_path=DB_FILE)
            
            # Clean and standardize the data
            combined_df = preprocessor.clean_and_standardize_data(current_df, marketed_df)
            
            # Create database schema
            preprocessor.create_database_schema()
            
            # Load data to database
            success = preprocessor.load_data_to_database(combined_df)
            
            if success:
                st.success("✅ Data successfully preprocessed and loaded into the database!")
                st.session_state.preprocess_done = True
                
                # Generate statistics
                stats = preprocessor._generate_summary_stats(combined_df)
                
                # Display comprehensive statistics
                st.subheader("📊 Processing Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Properties", len(combined_df))
                with col2:
                    st.metric("Current Properties", len(current_df))
                with col3:
                    st.metric("Marketed Properties", len(marketed_df))
                
                # Show detailed statistics
                if stats:
                    st.subheader("📈 Dataset Statistics")
                    
                    # Regional distribution
                    if 'regions' in stats and stats['regions']:
                        st.write("**Regional Distribution:**")
                        region_df = pd.DataFrame(list(stats['regions'].items()), 
                                               columns=['Region', 'Count'])
                        st.dataframe(region_df, use_container_width=True)
                    
                    # Key metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'avg_size_sqm' in stats:
                            st.metric("Average Size", f"{stats['avg_size_sqm']:,.0f} sqm")
                    with col2:
                        if 'avg_build_year' in stats:
                            st.metric("Average Build Year", f"{stats['avg_build_year']:.0f}")
                    
                    # Data quality metrics
                    if 'properties_with_coordinates' in stats:
                        coord_pct = (stats['properties_with_coordinates'] / stats['total_properties']) * 100
                        st.metric("Properties with Coordinates", 
                                 f"{stats['properties_with_coordinates']} ({coord_pct:.1f}%)")
                
                # Show sample data
                st.subheader("🔍 Data Preview")
                st.dataframe(combined_df.head(5), use_container_width=True)
                
                # Database information
                st.subheader("🗄️ Database Information")
                st.info(f"**Database Location:** `{preprocessor.db_path}`")
                st.write("The database includes optimized indexes for:")
                st.write("- Marketing status (is_marketed)")
                st.write("- Geographic coordinates (latitude, longitude)")
                st.write("- Property size and build year")
                st.write("- Regional distribution")
            else:
                st.error("❌ Failed to load data to database")
                st.session_state.preprocess_done = False
            
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
        st.write("**Error Details:**")
        st.code(str(e))
        
        st.write("**Please check:**")
        st.write("- Excel files are present in the `data/` directory")
        st.write("- Database directory has write permissions")
        st.write("- All required dependencies are installed")
        st.session_state.preprocess_done = False

# Add information section
st.subheader("ℹ️ About the Data")
st.write("""
**Current Portfolio (1,250 properties):**
- Industrial properties across UK regions
- Complete property details including size, location, build year
- Physical characteristics (eaves height, parking, doors)
- EPC ratings and regional classifications

**Marketed Warehouses (5 properties):**
- Cherry Lane (Unit 10, North West)
- Tech Hub (Unit 8, South East)
- Spitfire Park (Unit 42, Midlands)
- Stable Lane (Unit 14, South East)
- Chancery Depot (Unit 2)

**Database Schema:**
The processed data is stored in a SQLite database with optimized schema for:
- Fast similarity searches
- Geographic proximity queries
- Regional and size-based filtering
- Correlation analysis across property characteristics
""")
