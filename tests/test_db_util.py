
import unittest
import pandas as pd
from utils import db_util

class TestDbUtil(unittest.TestCase):

    def setUp(self):
        # Use an in-memory SQLite database for testing
        self.engine = db_util.create_engine("sqlite:///:memory:")
        db_util.ENGINE = self.engine
        db_util.create_tables()

        data = {
            'property_id': [1, 2],
            'industrial_estate_name': ['Test Estate 1', 'Test Estate 2'],
            'unit_name': ['Unit 1', 'Unit 2'],
            'region': ['Region 1', 'Region 2'],
            'latitude': [51.5, 52.5],
            'longitude': [-0.1, -0.2],
            'car_parking_spaces': [10, 20],
            'size_sqm': [1000, 2000],
            'build_year': [2000, 2010],
            'yard_depth_m': [10, 20],
            'min_eaves_m': [5, 6],
            'max_eaves_m': [6, 7],
            'doors': [2, 4],
            'epc_rating': ['A', 'B'],
            'is_marketed': [False, True]
        }
        self.df = pd.DataFrame(data)
        db_util.load_df_to_db(self.df, "properties")

    def test_query_db(self):
        result_df = db_util.query_db("SELECT * FROM properties")
        self.assertEqual(len(result_df), 2)
        self.assertEqual(result_df['industrial_estate_name'][0], 'Test Estate 1')

if __name__ == '__main__':
    unittest.main()

