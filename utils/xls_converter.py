'''
Utility for loading and preprocessing data from Excel files.
'''
import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the dataframe by cleaning and imputing missing values.
    """
    # Fill missing geo with 0
    df["Latitude"] = df["Latitude"].fillna(0)
    df["Longitude"] = df["Longitude"].fillna(0)

    # Impute missing Build Year with the median
    if df["Build Year"].isnull().any():
        median_build_year = df["Build Year"].median()
        df["Build Year"] = df["Build Year"].fillna(median_build_year)

    # Clean and convert numeric columns
    for col in ["Yard Depth m", "Min. Eaves m", "Max. Eaves m", "Doors #"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Fill missing EPC Rating
    df["EPC Rating"] = df["EPC Rating"].fillna("Not Available")

    # Convert data types
    df["Build Year"] = df["Build Year"].astype(int)
    df["Car Parking Spaces #"] = df["Car Parking Spaces #"].astype(int)
    df["Doors #"] = df["Doors #"].astype(int)

    return df

def load_current_portfolio(path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the Current Portfolio data.
    """
    df = pd.read_excel(path)
    df = preprocess_data(df)
    return df

def load_marketed_warehouses(path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the Marketed Warehouses data.
    """
    df = pd.read_excel(path)
    df = preprocess_data(df)
    return df

