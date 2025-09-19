"""
Streamlit page for preprocessing the MLI property data.
"""
import streamlit as st
import pandas as pd
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocess_util import run_preprocessing
from utils.streamlit_db_util import DB_FILE, check_db_initialized

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

# Check if data is already loaded
if check_db_initialized():
    st.success("✅ Data is already loaded in the database.")
    
    # Option to reload data
    if st.button("Reload Data"):
        st.session_state.preprocess_done = False

# Function to get the absolute path to data files
def get_data_path(filename):
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root
    project_root = os.path.dirname(current_dir)
    # Return the path to the data file
    return os.path.join(project_root, "data", filename)

if st.button("Run Preprocessing", type="primary") or st.session_state.get('preprocess_done') == False:
    try:
        with st.spinner("Loading and processing data..."):
            # Get paths to data files
            current_portfolio_path = get_data_path("CurrentPortfolio.xlsx")
            marketed_warehouses_path = get_data_path("MarketedWarehouses.xlsx")
            
            # Run the complete preprocessing pipeline with Streamlit-specific DB path
            result = run_preprocessing(
                current_portfolio_path=current_portfolio_path,
                marketed_warehouses_path=marketed_warehouses_path,
                db_path=DB_FILE
            )
            
        if result['success']:
            st.success("✅ Data successfully preprocessed and loaded into the database!")
            st.session_state.preprocess_done = True
            
            # Display comprehensive statistics
            st.subheader("📊 Processing Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Properties", result['total_properties'])
            with col2:
                st.metric("Current Properties", result['current_properties'])
            with col3:
                st.metric("Marketed Properties", result['marketed_properties'])
            
            # Show detailed statistics
            if result['statistics']:
                st.subheader("📈 Dataset Statistics")
                
                stats = result['statistics']
                
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
            if result['sample_data']:
                st.subheader("🔍 Data Preview")
                sample_df = pd.DataFrame(result['sample_data'])
                st.dataframe(sample_df, use_container_width=True)
            
            # Database information
            st.subheader("🗄️ Database Information")
            st.info(f"**Database Location:** `{result['database_path']}`")
            st.write("The database includes optimized indexes for:")
            st.write("- Marketing status (is_marketed)")
            st.write("- Geographic coordinates (latitude, longitude)")
            st.write("- Property size and build year")
            st.write("- Regional distribution")
            
        else:
            st.error(f"❌ Preprocessing failed: {result['error']}")
            st.write("**Troubleshooting Tips:**")
            st.write("1. Ensure Excel files exist in the `data/` directory")
            st.write("2. Check file permissions for database creation")
            st.write("3. Verify sufficient disk space for database")
            st.write("4. Check the error message above for specific details")
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
