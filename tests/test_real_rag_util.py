#!/usr/bin/env python3
"""
Unit tests for REAL RAG utility functionality with actual OpenAI embeddings.
Uses real vector database and OpenAI API - no mock data.
"""

import unittest
import json
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.real_rag_util import RealRAGUtil
from utils import db_util


class TestRealRAGUtil(unittest.TestCase):
    """Test cases for Real RAG utility functionality with actual embeddings."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.test_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test-data')
        os.makedirs(cls.test_data_dir, exist_ok=True)
        
        # Check if we have OpenAI API key
        cls.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not cls.openai_api_key:
            raise unittest.SkipTest("OpenAI API key not found - skipping real RAG tests")
        
        # Initialize RAG utility once for all tests
        print("\n🚀 Initializing Real RAG Utility with OpenAI embeddings...")
        cls.rag_util = RealRAGUtil(openai_api_key=cls.openai_api_key)
        
        # Initialize vectors once
        success = cls.rag_util.initialize_vectors()
        if not success:
            raise unittest.SkipTest("Failed to initialize vectors - skipping real RAG tests")
        
        print(f"✅ RAG utility initialized with {len(cls.rag_util.property_data)} properties")
    
    def test_01_vector_initialization(self):
        """Test RAG-1: Real vector initialization with OpenAI embeddings."""
        print("\n🔍 Testing Real RAG Vector Initialization")
        
        # Verify initialization was successful
        self.assertIsNotNone(self.rag_util.property_data, "Property data should be loaded")
        self.assertIsNotNone(self.rag_util.property_vectors, "Property vectors should be created")
        self.assertIsNotNone(self.rag_util.faiss_index, "FAISS index should be created")
        
        # Validate vector dimensions (OpenAI text-embedding-3-small = 1536 dimensions)
        expected_properties = len(self.rag_util.property_data)
        expected_dim = 1536
        
        self.assertEqual(self.rag_util.property_vectors.shape[0], expected_properties)
        self.assertEqual(self.rag_util.property_vectors.shape[1], expected_dim)
        
        # Validate FAISS index
        self.assertEqual(self.rag_util.faiss_index.ntotal, expected_properties)
        
        # Check for marketed properties
        marketed_count = int((self.rag_util.property_data['is_marketed'] == 1).sum())
        
        # Save test results
        result_data = {
            "test": "real_vector_initialization",
            "success": True,
            "properties_loaded": expected_properties,
            "marketed_properties": marketed_count,
            "vector_dimensions": expected_dim,
            "vector_shape": list(self.rag_util.property_vectors.shape),
            "faiss_index_size": int(self.rag_util.faiss_index.ntotal),
            "embedding_model": self.rag_util.vectorizer.embedding_model,
            "sample_vector_stats": {
                "mean": float(np.mean(self.rag_util.property_vectors[0])),
                "std": float(np.std(self.rag_util.property_vectors[0])),
                "min": float(np.min(self.rag_util.property_vectors[0])),
                "max": float(np.max(self.rag_util.property_vectors[0])),
                "norm": float(np.linalg.norm(self.rag_util.property_vectors[0]))
            },
            "error": None
        }
        
        with open(os.path.join(self.test_data_dir, 'real_rag_vector_initialization.json'), 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"✅ Real vector initialization completed")
        print(f"   Properties loaded: {expected_properties}")
        print(f"   Marketed properties: {marketed_count}")
        print(f"   Vector dimensions: {expected_dim}")
        print(f"   FAISS index size: {self.rag_util.faiss_index.ntotal}")
    
    def test_02_similarity_search_by_id(self):
        """Test RAG-2: Real property similarity search by property ID."""
        print("\n🔍 Testing Real RAG Similarity Search by Property ID")
        
        # Get a sample marketed property ID if available
        marketed_properties = self.rag_util.property_data[
            self.rag_util.property_data['is_marketed'] == 1
        ]
        
        if len(marketed_properties) > 0:
            sample_property_id = marketed_properties.iloc[0]['property_id']
            print(f"   Using marketed property ID: {sample_property_id}")
        else:
            # Use first available property
            sample_property_id = self.rag_util.property_data.iloc[0]['property_id']
            print(f"   Using first property ID: {sample_property_id}")
        
        # Test similarity search
        result = self.rag_util.find_similar_properties(
            target_property_id=sample_property_id,
            top_k=10
        )
        
        self.assertTrue(result['success'], f"Similarity search should succeed: {result.get('error')}")
        self.assertEqual(result['query_type'], 'similarity_search')
        self.assertEqual(len(result['results']), 10, "Should return 10 similar properties")
        self.assertEqual(result['target_property_id'], sample_property_id)
        
        # Validate similarity scores (should be between 0 and 1 for cosine similarity)
        similarities = [prop['similarity_score'] for prop in result['results']]
        self.assertTrue(all(0 <= score <= 1 for score in similarities), 
                       "Similarity scores should be between 0 and 1")
        self.assertTrue(similarities == sorted(similarities, reverse=True), 
                       "Results should be sorted by similarity score")
        
        # First result should be the target property itself (or very similar)
        self.assertGreater(similarities[0], 0.95, "Top similarity should be very high")
        
        # Save results
        with open(os.path.join(self.test_data_dir, 'real_rag_similarity_by_id.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Real similarity search by ID completed")
        print(f"   Target property ID: {sample_property_id}")
        print(f"   Similar properties found: {len(result['results'])}")
        print(f"   Top similarity score: {similarities[0]:.3f}")
        print(f"   Average similarity: {result['avg_similarity']:.3f}")
    
    def test_03_similarity_search_by_query(self):
        """Test RAG-3: Real property similarity search by text query."""
        print("\n🔍 Testing Real RAG Similarity Search by Text Query")
        
        # Test with a descriptive query
        query_text = "Large industrial warehouse with good parking facilities in the Midlands region"
        
        # Test similarity search
        result = self.rag_util.find_similar_properties(
            query_text=query_text,
            top_k=8
        )
        
        self.assertTrue(result['success'], f"Query-based similarity search should succeed: {result.get('error')}")
        self.assertEqual(result['query_type'], 'similarity_search')
        self.assertEqual(len(result['results']), 8, "Should return 8 similar properties")
        self.assertEqual(result['query_text'], query_text)
        
        # Validate results structure
        for prop in result['results']:
            self.assertIn('similarity_score', prop)
            self.assertIn('industrial_estate_name', prop)
            self.assertIn('size_sqm', prop)
            self.assertIn('region', prop)
            self.assertIn('rank', prop)
        
        # Check if results are relevant to the query (should have some Midlands properties)
        regions = [prop.get('region', '') for prop in result['results']]
        print(f"   Regions found: {set(regions)}")
        
        # Save results
        with open(os.path.join(self.test_data_dir, 'real_rag_similarity_by_query.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Real similarity search by query completed")
        print(f"   Query: {query_text}")
        print(f"   Similar properties found: {len(result['results'])}")
        print(f"   Average similarity: {result['avg_similarity']:.3f}")
    
    def test_04_cluster_analysis(self):
        """Test RAG-4: Real property cluster analysis."""
        print("\n🔍 Testing Real RAG Property Cluster Analysis")
        
        # Test cluster analysis
        result = self.rag_util.analyze_property_clusters(n_clusters=6)
        
        self.assertTrue(result['success'], f"Cluster analysis should succeed: {result.get('error')}")
        self.assertEqual(result['query_type'], 'cluster_analysis')
        self.assertEqual(result['n_clusters'], 6)
        self.assertEqual(len(result['cluster_summary']), 6)
        
        # Validate cluster summary
        total_clustered = 0
        for cluster_id, cluster_info in result['cluster_summary'].items():
            self.assertIn('size', cluster_info)
            self.assertIn('percentage', cluster_info)
            self.assertIn('avg_size_sqm', cluster_info)
            self.assertIn('avg_build_year', cluster_info)
            self.assertIn('regions', cluster_info)
            self.assertIn('marketed_count', cluster_info)
            self.assertIn('sample_properties', cluster_info)
            self.assertGreater(cluster_info['size'], 0)
            total_clustered += cluster_info['size']
        
        self.assertEqual(total_clustered, result['total_properties'])
        
        # Find largest and smallest clusters
        cluster_sizes = [info['size'] for info in result['cluster_summary'].values()]
        largest_cluster = max(cluster_sizes)
        smallest_cluster = min(cluster_sizes)
        
        # Save results
        with open(os.path.join(self.test_data_dir, 'real_rag_cluster_analysis.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Real cluster analysis completed")
        print(f"   Number of clusters: {result['n_clusters']}")
        print(f"   Total properties clustered: {total_clustered}")
        print(f"   Largest cluster size: {largest_cluster}")
        print(f"   Smallest cluster size: {smallest_cluster}")
    
    def test_05_homogeneity_analysis(self):
        """Test RAG-5: Real portfolio homogeneity analysis."""
        print("\n🔍 Testing Real RAG Portfolio Homogeneity Analysis")
        
        # Test homogeneity analysis
        result = self.rag_util.calculate_portfolio_homogeneity()
        
        self.assertTrue(result['success'], f"Homogeneity analysis should succeed: {result.get('error')}")
        self.assertEqual(result['query_type'], 'homogeneity_analysis')
        
        # Validate required metrics
        required_metrics = [
            'overall_homogeneity', 'homogeneity_std', 'marketed_vs_portfolio',
            'marketed_internal_homogeneity', 'portfolio_internal_homogeneity',
            'homogeneity_coefficient'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, result)
            self.assertIsInstance(result[metric], (int, float))
        
        # Validate value ranges for cosine similarity
        self.assertGreaterEqual(result['overall_homogeneity'], -1)
        self.assertLessEqual(result['overall_homogeneity'], 1)
        self.assertGreaterEqual(result['homogeneity_std'], 0)
        
        # Validate similarity distribution
        self.assertIn('similarity_distribution', result)
        dist = result['similarity_distribution']
        self.assertIn('min', dist)
        self.assertIn('max', dist)
        self.assertIn('median', dist)
        
        # Save results
        with open(os.path.join(self.test_data_dir, 'real_rag_homogeneity_analysis.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Real homogeneity analysis completed")
        print(f"   Overall homogeneity: {result['overall_homogeneity']:.3f}")
        print(f"   Marketed vs portfolio: {result['marketed_vs_portfolio']:.3f}")
        print(f"   Homogeneity coefficient: {result['homogeneity_coefficient']:.3f}")
        print(f"   Similarity range: {dist['min']:.3f} to {dist['max']:.3f}")
    
    def test_06_comprehensive_real_rag_pipeline(self):
        """Test RAG-6: Comprehensive real RAG analysis pipeline."""
        print("\n🔍 Testing Comprehensive Real RAG Analysis Pipeline")
        
        # Run all RAG analyses with real data
        similarity_result = self.rag_util.find_similar_properties(
            query_text="Modern industrial facility with loading docks",
            top_k=5
        )
        cluster_result = self.rag_util.analyze_property_clusters(n_clusters=4)
        homogeneity_result = self.rag_util.calculate_portfolio_homogeneity()
        
        # Combine results into comprehensive analysis
        comprehensive_result = {
            "test": "comprehensive_real_rag_pipeline",
            "success": all([
                similarity_result.get('success', False),
                cluster_result.get('success', False),
                homogeneity_result.get('success', False)
            ]),
            "analysis_timestamp": "2025-09-19",
            "embedding_model": self.rag_util.vectorizer.embedding_model,
            "portfolio_overview": {
                "total_properties": len(self.rag_util.property_data),
                "vector_dimensions": self.rag_util.property_vectors.shape[1],
                "marketed_properties": int((self.rag_util.property_data['is_marketed'] == 1).sum()),
                "faiss_index_size": int(self.rag_util.faiss_index.ntotal)
            },
            "similarity_analysis": {
                "success": similarity_result.get('success', False),
                "query_text": similarity_result.get('query_text'),
                "top_similar_properties": len(similarity_result.get('results', [])),
                "avg_similarity_score": similarity_result.get('avg_similarity', 0),
                "max_similarity_score": similarity_result.get('max_similarity', 0),
                "top_matches": similarity_result.get('results', [])[:3] if similarity_result.get('success') else []
            },
            "cluster_analysis": {
                "success": cluster_result.get('success', False),
                "n_clusters": cluster_result.get('n_clusters', 0),
                "largest_cluster_size": max(
                    (c.get('size', 0) for c in cluster_result.get('cluster_summary', {}).values()),
                    default=0
                ),
                "cluster_distribution": {
                    f"cluster_{i}": cluster_result.get('cluster_summary', {}).get(f'cluster_{i}', {}).get('size', 0)
                    for i in range(cluster_result.get('n_clusters', 0))
                }
            },
            "homogeneity_analysis": {
                "success": homogeneity_result.get('success', False),
                "overall_homogeneity": homogeneity_result.get('overall_homogeneity', 0),
                "marketed_vs_portfolio": homogeneity_result.get('marketed_vs_portfolio', 0),
                "homogeneity_coefficient": homogeneity_result.get('homogeneity_coefficient', 0),
                "similarity_range": {
                    "min": homogeneity_result.get('similarity_distribution', {}).get('min', 0),
                    "max": homogeneity_result.get('similarity_distribution', {}).get('max', 0),
                    "median": homogeneity_result.get('similarity_distribution', {}).get('median', 0)
                }
            },
            "insights": {
                "portfolio_diversity_score": 1 - homogeneity_result.get('overall_homogeneity', 0),
                "embedding_quality": "high" if similarity_result.get('max_similarity', 0) > 0.8 else "medium",
                "cluster_balance": "balanced" if max(
                    (c.get('size', 0) for c in cluster_result.get('cluster_summary', {}).values()),
                    default=0
                ) < len(self.rag_util.property_data) * 0.4 else "unbalanced"
            },
            "performance_metrics": {
                "vector_initialization_success": True,
                "faiss_search_enabled": True,
                "openai_embeddings_used": True,
                "total_embedding_dimensions": self.rag_util.property_vectors.shape[1]
            },
            "error": None
        }
        
        # Validate comprehensive results
        self.assertTrue(similarity_result.get('success', False), 
                       f"Similarity analysis failed: {similarity_result.get('error')}")
        self.assertTrue(cluster_result.get('success', False), 
                       f"Cluster analysis failed: {cluster_result.get('error')}")
        self.assertTrue(homogeneity_result.get('success', False), 
                       f"Homogeneity analysis failed: {homogeneity_result.get('error')}")
        self.assertTrue(comprehensive_result['success'], "Overall comprehensive analysis should succeed")
        
        # Save comprehensive results
        with open(os.path.join(self.test_data_dir, 'real_rag_comprehensive_pipeline.json'), 'w') as f:
            json.dump(comprehensive_result, f, indent=2)
        
        print(f"✅ Comprehensive real RAG pipeline completed")
        print(f"   Portfolio size: {comprehensive_result['portfolio_overview']['total_properties']}")
        print(f"   Embedding model: {comprehensive_result['embedding_model']}")
        print(f"   Similarity analysis: {comprehensive_result['similarity_analysis']['success']}")
        print(f"   Cluster analysis: {comprehensive_result['cluster_analysis']['success']}")
        print(f"   Homogeneity analysis: {comprehensive_result['homogeneity_analysis']['success']}")
        print(f"   Portfolio diversity: {comprehensive_result['insights']['portfolio_diversity_score']:.3f}")


if __name__ == '__main__':
    print("=" * 70)
    print("🧪 RUNNING REAL RAG UTILITY UNIT TESTS (WITH OPENAI EMBEDDINGS)")
    print("=" * 70)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 70)
    print("✅ REAL RAG UTILITY TESTS COMPLETED")
    print("=" * 70)
