#!/usr/bin/env python3
"""
Unit tests for RAG chat functionality with vectorized property analysis.
"""

import unittest
import json
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.rag_chat import MockRAGChat
from utils import db_util


class TestRAGChat(unittest.TestCase):
    """Test cases for RAG chat functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.rag_chat = MockRAGChat()
        self.test_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test-data')
        os.makedirs(self.test_data_dir, exist_ok=True)
    
    def test_vector_initialization(self):
        """Test vector initialization and property data loading."""
        print("\n🔍 Testing Vector Initialization")
        
        success = self.rag_chat.initialize_vectors()
        self.assertTrue(success, "Vector initialization should succeed")
        
        self.assertIsNotNone(self.rag_chat.property_data, "Property data should be loaded")
        self.assertIsNotNone(self.rag_chat.property_vectors, "Property vectors should be created")
        
        print(f"✅ Vectors initialized successfully")
        print(f"   Properties loaded: {len(self.rag_chat.property_data)}")
        print(f"   Vector dimensions: {self.rag_chat.property_vectors.shape}")
    
    def test_similarity_search(self):
        """Test property similarity search functionality."""
        print("\n🔍 Testing Property Similarity Search")
        
        # Initialize vectors
        self.rag_chat.initialize_vectors()
        
        # Test similarity search
        similar_props = self.rag_chat.find_similar_properties(top_k=10)
        
        self.assertIsInstance(similar_props, pd.DataFrame, "Should return DataFrame")
        self.assertLessEqual(len(similar_props), 10, "Should return at most 10 properties")
        self.assertIn('similarity_score', similar_props.columns, "Should include similarity scores")
        
        # Validate similarity scores
        scores = similar_props['similarity_score']
        self.assertTrue(all(0 <= score <= 1 for score in scores), "Similarity scores should be between 0 and 1")
        self.assertTrue(scores.is_monotonic_decreasing, "Scores should be sorted in descending order")
        
        # Save results
        result_data = {
            "test": "similarity_search",
            "success": True,
            "properties_found": len(similar_props),
            "top_similarity_score": float(scores.iloc[0]),
            "avg_similarity_score": float(scores.mean()),
            "sample_properties": similar_props.head(3)[['industrial_estate_name', 'unit_name', 'region', 'size_sqm', 'similarity_score']].to_dict('records'),
            "error": None
        }
        
        with open(os.path.join(self.test_data_dir, 'rag_similarity_search.json'), 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"✅ Similarity search completed")
        print(f"   Properties found: {len(similar_props)}")
        print(f"   Top similarity score: {scores.iloc[0]:.3f}")
        print(f"   Average similarity: {scores.mean():.3f}")
    
    def test_cluster_analysis(self):
        """Test property cluster analysis."""
        print("\n🔍 Testing Property Cluster Analysis")
        
        # Initialize vectors
        self.rag_chat.initialize_vectors()
        
        # Test cluster analysis
        cluster_results = self.rag_chat.analyze_property_clusters(n_clusters=5)
        
        self.assertIsInstance(cluster_results, dict, "Should return dictionary")
        self.assertEqual(cluster_results['n_clusters'], 5, "Should have 5 clusters")
        self.assertIn('cluster_summary', cluster_results, "Should include cluster summary")
        
        # Validate cluster summary
        cluster_summary = cluster_results['cluster_summary']
        self.assertEqual(len(cluster_summary), 5, "Should have 5 cluster summaries")
        
        for cluster_id, cluster_info in cluster_summary.items():
            self.assertIn('size', cluster_info, f"Cluster {cluster_id} should have size")
            self.assertIn('avg_size_sqm', cluster_info, f"Cluster {cluster_id} should have avg_size_sqm")
            self.assertIn('avg_build_year', cluster_info, f"Cluster {cluster_id} should have avg_build_year")
            self.assertGreater(cluster_info['size'], 0, f"Cluster {cluster_id} should have positive size")
        
        # Save results
        with open(os.path.join(self.test_data_dir, 'rag_cluster_analysis.json'), 'w') as f:
            json.dump(cluster_results, f, indent=2)
        
        total_properties = sum(cluster['size'] for cluster in cluster_summary.values())
        print(f"✅ Cluster analysis completed")
        print(f"   Number of clusters: {cluster_results['n_clusters']}")
        print(f"   Total properties clustered: {total_properties}")
        print(f"   Largest cluster size: {max(cluster['size'] for cluster in cluster_summary.values())}")
    
    def test_homogeneity_analysis(self):
        """Test portfolio homogeneity analysis."""
        print("\n🔍 Testing Portfolio Homogeneity Analysis")
        
        # Initialize vectors
        self.rag_chat.initialize_vectors()
        
        # Test homogeneity analysis
        homogeneity_results = self.rag_chat.calculate_portfolio_homogeneity()
        
        self.assertIsInstance(homogeneity_results, dict, "Should return dictionary")
        
        # Validate required metrics
        required_metrics = [
            'overall_homogeneity',
            'homogeneity_std', 
            'marketed_vs_portfolio',
            'marketed_internal_homogeneity',
            'homogeneity_coefficient'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, homogeneity_results, f"Should include {metric}")
            self.assertIsInstance(homogeneity_results[metric], (int, float), f"{metric} should be numeric")
        
        # Validate value ranges
        self.assertGreaterEqual(homogeneity_results['overall_homogeneity'], 0, "Overall homogeneity should be non-negative")
        self.assertLessEqual(homogeneity_results['overall_homogeneity'], 1, "Overall homogeneity should be <= 1")
        
        # Save results
        with open(os.path.join(self.test_data_dir, 'rag_homogeneity_analysis.json'), 'w') as f:
            json.dump(homogeneity_results, f, indent=2)
        
        print(f"✅ Homogeneity analysis completed")
        print(f"   Overall homogeneity: {homogeneity_results['overall_homogeneity']:.3f}")
        print(f"   Marketed vs portfolio: {homogeneity_results['marketed_vs_portfolio']:.3f}")
        print(f"   Homogeneity coefficient: {homogeneity_results['homogeneity_coefficient']:.3f}")
    
    def test_rag_query_similarity(self):
        """Test RAG query for similarity analysis."""
        print("\n🔍 Testing RAG Query: Property Similarity")
        
        query = "Find the most similar properties to the marketed warehouses in our portfolio"
        result = self.rag_chat.query_with_context(query)
        
        self.assertIsInstance(result, dict, "Should return dictionary")
        self.assertTrue(result['success'], "Query should succeed")
        self.assertEqual(result['analysis_type'], 'similarity_search', "Should be similarity search")
        self.assertIsNotNone(result['response'], "Should have response")
        self.assertIsNotNone(result['context_data'], "Should have context data")
        
        # Save results
        with open(os.path.join(self.test_data_dir, 'rag_query_similarity.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Similarity query completed")
        print(f"   Query: {query}")
        print(f"   Analysis type: {result['analysis_type']}")
        print(f"   Success: {result['success']}")
    
    def test_rag_query_clustering(self):
        """Test RAG query for cluster analysis."""
        print("\n🔍 Testing RAG Query: Property Clustering")
        
        query = "Analyze the property clusters in our portfolio and identify distinct groups"
        result = self.rag_chat.query_with_context(query)
        
        self.assertIsInstance(result, dict, "Should return dictionary")
        self.assertTrue(result['success'], "Query should succeed")
        self.assertEqual(result['analysis_type'], 'cluster_analysis', "Should be cluster analysis")
        self.assertIsNotNone(result['response'], "Should have response")
        self.assertIsNotNone(result['context_data'], "Should have context data")
        
        # Save results
        with open(os.path.join(self.test_data_dir, 'rag_query_clustering.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Clustering query completed")
        print(f"   Query: {query}")
        print(f"   Analysis type: {result['analysis_type']}")
        print(f"   Success: {result['success']}")
    
    def test_rag_query_homogeneity(self):
        """Test RAG query for homogeneity analysis."""
        print("\n🔍 Testing RAG Query: Portfolio Homogeneity")
        
        query = "What is the homogeneity score of our marketed properties compared to the rest of the portfolio?"
        result = self.rag_chat.query_with_context(query)
        
        self.assertIsInstance(result, dict, "Should return dictionary")
        self.assertTrue(result['success'], "Query should succeed")
        self.assertEqual(result['analysis_type'], 'homogeneity_analysis', "Should be homogeneity analysis")
        self.assertIsNotNone(result['response'], "Should have response")
        self.assertIsNotNone(result['context_data'], "Should have context data")
        
        # Save results
        with open(os.path.join(self.test_data_dir, 'rag_query_homogeneity.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Homogeneity query completed")
        print(f"   Query: {query}")
        print(f"   Analysis type: {result['analysis_type']}")
        print(f"   Success: {result['success']}")
    
    def test_comprehensive_rag_analysis(self):
        """Test comprehensive RAG analysis combining all features."""
        print("\n🔍 Testing Comprehensive RAG Analysis")
        
        # Initialize vectors
        self.rag_chat.initialize_vectors()
        
        # Run all analyses
        similarity_results = self.rag_chat.find_similar_properties(top_k=5)
        cluster_results = self.rag_chat.analyze_property_clusters(n_clusters=3)
        homogeneity_results = self.rag_chat.calculate_portfolio_homogeneity()
        
        # Combine results
        comprehensive_results = {
            "analysis_timestamp": "2025-09-18",
            "portfolio_overview": {
                "total_properties": len(self.rag_chat.property_data),
                "vector_dimensions": self.rag_chat.property_vectors.shape[1] if self.rag_chat.property_vectors is not None else 0
            },
            "similarity_analysis": {
                "top_similar_properties": len(similarity_results),
                "avg_similarity_score": float(similarity_results['similarity_score'].mean()),
                "top_matches": similarity_results.head(3)[['industrial_estate_name', 'unit_name', 'similarity_score']].to_dict('records')
            },
            "cluster_analysis": cluster_results,
            "homogeneity_analysis": homogeneity_results,
            "insights": {
                "most_homogeneous_metric": max(homogeneity_results.keys(), key=lambda k: homogeneity_results[k]),
                "largest_cluster_size": max(cluster['size'] for cluster in cluster_results['cluster_summary'].values()),
                "portfolio_diversity_score": 1 - homogeneity_results['overall_homogeneity']
            }
        }
        
        # Save comprehensive results
        with open(os.path.join(self.test_data_dir, 'rag_comprehensive_analysis.json'), 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        print(f"✅ Comprehensive analysis completed")
        print(f"   Portfolio size: {comprehensive_results['portfolio_overview']['total_properties']}")
        print(f"   Average similarity: {comprehensive_results['similarity_analysis']['avg_similarity_score']:.3f}")
        print(f"   Number of clusters: {comprehensive_results['cluster_analysis']['n_clusters']}")
        print(f"   Overall homogeneity: {comprehensive_results['homogeneity_analysis']['overall_homogeneity']:.3f}")


if __name__ == '__main__':
    print("=" * 60)
    print("🧪 RUNNING RAG CHAT UNIT TESTS")
    print("=" * 60)
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 60)
    print("✅ RAG CHAT TESTS COMPLETED")
    print("=" * 60)
