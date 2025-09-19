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
from utils.db_util import db

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

# Function to get the absolute path to data files
def get_data_path(filename):
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root
    project_root = os.path.dirname(current_dir)
    # Return the path to the data file
    return os.path.join(project_root, "data", filename)

# Initialize session state for preprocessing status
if 'preprocess_done' not in st.session_state:
    st.session_state.preprocess_done = db.check_initialized()

# Check if data is already loaded
if st.session_state.preprocess_done:
    st.success("✅ Data is already loaded in the database.")
    
    # Option to reload data
    if st.button("Reload Data"):
        st.session_state.preprocess_done = False

if st.button("Run Preprocessing", type="primary") or st.session_state.get('preprocess_done') == False:
    try:
        with st.spinner("Loading and processing data..."):
            # Get paths to data files
            current_portfolio_path = get_data_path("CurrentPortfolio.xlsx")
            marketed_warehouses_path = get_data_path("MarketedWarehouses.xlsx")
            
            # Load Excel files
            try:
                current_df = pd.read_excel(current_portfolio_path)
                marketed_df = pd.read_excel(marketed_warehouses_path)
                st.info(f"Loaded {len(current_df)} current properties and {len(marketed_df)} marketed properties")
            except Exception as e:
                st.error(f"Error loading Excel files: {e}")
                st.stop()
            
            # Initialize preprocessor with the database path
            preprocessor = PropertyPreprocessor(db_path=db.db_path)
            
            # Clean and standardize the data
            combined_df = preprocessor.clean_and_standardize_data(current_df, marketed_df)
            
            # Create database schema
            db.create_tables()
            
            # Load data to database
            success = db.load_dataframe(combined_df)
            
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
                st.info(f"**Database Location:** `{db.db_path}`")
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
