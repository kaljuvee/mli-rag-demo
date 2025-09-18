'''
Tests for the xls_converter utility.
'''
import unittest
import pandas as pd
from utils import xls_converter

class TestXlsConverter(unittest.TestCase):

    def test_preprocess_data(self):
        data = {
            'Latitude': [None, 1.0],
            'Longitude': [None, 1.0],
            'Build Year': [2000, None],
            'Yard Depth m': ['10', None],
            'Min. Eaves m': ['5', None],
            'Max. Eaves m': ['6', None],
            'Doors #': ['2', None],
            'EPC Rating': ['A', None],
            'Car Parking Spaces #': [10, 20]
        }
        df = pd.DataFrame(data)
        processed_df = xls_converter.preprocess_data(df)

        self.assertEqual(processed_df['Latitude'].isnull().sum(), 0)
        self.assertEqual(processed_df['Longitude'].isnull().sum(), 0)
        self.assertEqual(processed_df['Build Year'].isnull().sum(), 0)
        self.assertEqual(processed_df['Yard Depth m'].isnull().sum(), 0)
        self.assertEqual(processed_df['Min. Eaves m'].isnull().sum(), 0)
        self.assertEqual(processed_df['Max. Eaves m'].isnull().sum(), 0)
        self.assertEqual(processed_df['Doors #'].isnull().sum(), 0)
        self.assertEqual(processed_df['EPC Rating'].isnull().sum(), 0)

if __name__ == '__main__':
    unittest.main()

