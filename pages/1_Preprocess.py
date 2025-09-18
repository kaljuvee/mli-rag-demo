'''
Streamlit page for preprocessing the data.
'''
import streamlit as st
import pandas as pd
from utils import xls_converter, db_util

st.set_page_config(page_title="Preprocess Data", page_icon="⚙️")

st.title("Preprocess and Load Data")

if st.button("Run Preprocessing"):
    with st.spinner("Loading and preprocessing data..."):
        try:
            # Load and preprocess data
            current_portfolio_df = xls_converter.load_current_portfolio("data/CurrentPortfolio.xlsx")
            marketed_warehouses_df = xls_converter.load_marketed_warehouses("data/MarketedWarehouses.xlsx")

            # Add a flag to distinguish between the two datasets
            current_portfolio_df["is_marketed"] = False
            marketed_warehouses_df["is_marketed"] = True

            # Combine the dataframes
            combined_df = pd.concat([current_portfolio_df, marketed_warehouses_df], ignore_index=True)
            
            # Rename columns to be database-friendly
            combined_df.rename(columns={
                'Property Id': 'property_id',
                'Industrial Estate Name': 'industrial_estate_name',
                'Unit Name': 'unit_name',
                'Region': 'region',
                'Latitude': 'latitude',
                'Longitude': 'longitude',
                'Car Parking Spaces #': 'car_parking_spaces',
                'Size sqm': 'size_sqm',
                'Build Year': 'build_year',
                'Yard Depth m': 'yard_depth_m',
                'Min. Eaves m': 'min_eaves_m',
                'Max. Eaves m': 'max_eaves_m',
                'Doors #': 'doors',
                'EPC Rating': 'epc_rating'
            }, inplace=True)


            # Create database and table
            db_util.create_tables()

            # Load data into the database
            db_util.load_df_to_db(combined_df, "properties")

            st.success("Data successfully preprocessed and loaded into the database!")
            st.write("### Combined Data Preview")
            st.dataframe(combined_df.head())

        except Exception as e:
            st.error(f"An error occurred: {e}")

