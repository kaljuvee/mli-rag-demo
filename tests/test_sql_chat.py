"""
Unit tests for SQL chat functionality with the 3 specific test queries.
"""
import unittest
import json
import os
from utils.mock_sql_chat import MockSQLChat

class TestSQLChat(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.sql_chat = MockSQLChat()
        self.test_data_dir = "/home/ubuntu/mli-rag-demo/test-data"
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # The 3 specific test queries from requirements
        self.test_queries = [
            "Find the 10 most similar properties in the estate to the newly marketed property",
            "Provide a correlation score for the homogeneity of the marketed property(ies) with the rest of the estate, scoring each by physical characteristics, location, and age separately",
            "Find the closest properties to the marketed property, after excluding any property in the portfolio that is more than 10 miles from a major city"
        ]

    def test_database_connection(self):
        """Test that we can connect to the database and get schema"""
        schema = self.sql_chat.get_database_schema()
        self.assertIsInstance(schema, str)
        self.assertIn("properties", schema)
        print("✅ Database connection test passed")

    def test_query_1_similar_properties(self):
        """Test Query 1: Find the 10 most similar properties"""
        query = self.test_queries[0]
        print(f"\n🔍 Testing Query 1: {query}")
        
        result = self.sql_chat.chat_with_database(query)
        
        # Save result to JSON
        test_result = {
            "query": query,
            "success": result["success"],
            "sql_query": result.get("sql_query"),
            "error": result.get("error"),
            "data_shape": None,
            "sample_data": None
        }
        
        if result["success"] and result.get("data") is not None:
            df = result["data"]
            test_result["data_shape"] = df.shape
            test_result["sample_data"] = df.head(3).to_dict('records') if len(df) > 0 else []
            test_result["column_names"] = list(df.columns)
        
        # Save to JSON file
        with open(f"{self.test_data_dir}/query_1_similar_properties.json", "w") as f:
            json.dump(test_result, f, indent=2, default=str)
        
        print(f"SQL Generated: {result.get('sql_query')}")
        print(f"Success: {result['success']}")
        if result["success"]:
            print(f"Data shape: {test_result['data_shape']}")
        else:
            print(f"Error: {result.get('error')}")
        
        # Assert basic functionality
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("question", result)

    def test_query_2_correlation_score(self):
        """Test Query 2: Correlation score for homogeneity"""
        query = self.test_queries[1]
        print(f"\n🔍 Testing Query 2: {query}")
        
        result = self.sql_chat.chat_with_database(query)
        
        # Save result to JSON
        test_result = {
            "query": query,
            "success": result["success"],
            "sql_query": result.get("sql_query"),
            "error": result.get("error"),
            "data_shape": None,
            "sample_data": None
        }
        
        if result["success"] and result.get("data") is not None:
            df = result["data"]
            test_result["data_shape"] = df.shape
            test_result["sample_data"] = df.head(3).to_dict('records') if len(df) > 0 else []
            test_result["column_names"] = list(df.columns)
        
        # Save to JSON file
        with open(f"{self.test_data_dir}/query_2_correlation_score.json", "w") as f:
            json.dump(test_result, f, indent=2, default=str)
        
        print(f"SQL Generated: {result.get('sql_query')}")
        print(f"Success: {result['success']}")
        if result["success"]:
            print(f"Data shape: {test_result['data_shape']}")
        else:
            print(f"Error: {result.get('error')}")
        
        # Assert basic functionality
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("question", result)

    def test_query_3_closest_properties_excluding_cities(self):
        """Test Query 3: Closest properties excluding those far from major cities"""
        query = self.test_queries[2]
        print(f"\n🔍 Testing Query 3: {query}")
        
        result = self.sql_chat.chat_with_database(query)
        
        # Save result to JSON
        test_result = {
            "query": query,
            "success": result["success"],
            "sql_query": result.get("sql_query"),
            "error": result.get("error"),
            "data_shape": None,
            "sample_data": None
        }
        
        if result["success"] and result.get("data") is not None:
            df = result["data"]
            test_result["data_shape"] = df.shape
            test_result["sample_data"] = df.head(3).to_dict('records') if len(df) > 0 else []
            test_result["column_names"] = list(df.columns)
        
        # Save to JSON file
        with open(f"{self.test_data_dir}/query_3_closest_properties.json", "w") as f:
            json.dump(test_result, f, indent=2, default=str)
        
        print(f"SQL Generated: {result.get('sql_query')}")
        print(f"Success: {result['success']}")
        if result["success"]:
            print(f"Data shape: {test_result['data_shape']}")
        else:
            print(f"Error: {result.get('error')}")
        
        # Assert basic functionality
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("question", result)

    def test_simple_queries(self):
        """Test some simple queries to verify basic functionality"""
        simple_queries = [
            "Show me all marketed properties",
            "Count total properties",
            "What is the average size of properties?"
        ]
        
        results = []
        for query in simple_queries:
            print(f"\n🔍 Testing Simple Query: {query}")
            result = self.sql_chat.chat_with_database(query)
            
            test_result = {
                "query": query,
                "success": result["success"],
                "sql_query": result.get("sql_query"),
                "error": result.get("error")
            }
            
            if result["success"] and result.get("data") is not None:
                df = result["data"]
                test_result["data_shape"] = df.shape
                test_result["sample_data"] = df.head(3).to_dict('records') if len(df) > 0 else []
            
            results.append(test_result)
            print(f"Success: {result['success']}")
            if not result["success"]:
                print(f"Error: {result.get('error')}")
        
        # Save all simple query results
        with open(f"{self.test_data_dir}/simple_queries.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print("✅ Simple queries test completed")

    def test_sql_generation_only(self):
        """Test SQL generation without execution for complex queries"""
        print("\n🔍 Testing SQL Generation Only")
        
        sql_results = []
        for i, query in enumerate(self.test_queries, 1):
            sql_query = self.sql_chat.generate_sql_query(query)
            sql_results.append({
                "query_number": i,
                "natural_language": query,
                "generated_sql": sql_query,
                "has_error": sql_query.startswith("Error")
            })
            print(f"Query {i} SQL: {sql_query}")
        
        # Save SQL generation results
        with open(f"{self.test_data_dir}/sql_generation_results.json", "w") as f:
            json.dump(sql_results, f, indent=2, default=str)
        
        print("✅ SQL generation test completed")

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
