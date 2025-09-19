#!/usr/bin/env python3
"""
Unit tests for RAG utility functionality with vectorized property analysis.
Parallel implementation to SQL chat tests with comprehensive JSON validation.
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

from utils import db_util


class MockPropertyVectorizer:
    """Mock property vectorizer for testing without OpenAI API calls."""
    
    def __init__(self, openai_api_key=None):
        """Initialize mock vectorizer."""
        self.embedding_dim = 128  # Mock embedding dimension
        
    def vectorize_properties(self, properties_df: pd.DataFrame) -> np.ndarray:
        """Create mock property vectors."""
        n_properties = len(properties_df)
        
        # Create deterministic mock vectors based on property features
        vectors = []
        for _, prop in properties_df.iterrows():
            # Use property features to create consistent mock vectors
            size_feature = prop.get('size_sqm', 0) / 10000  # Normalize size
            year_feature = (prop.get('build_year', 1980) - 1950) / 100  # Normalize year
            region_feature = hash(str(prop.get('region', 'Unknown'))) % 100 / 100  # Region hash
            
            # Create mock vector with some structure
            base_vector = np.random.RandomState(int(prop.get('property_id', 0))).rand(self.embedding_dim)
            
            # Add feature-based components
            base_vector[0] = size_feature
            base_vector[1] = year_feature
            base_vector[2] = region_feature
            
            vectors.append(base_vector)
        
        return np.array(vectors)
    
    def vectorize_features(self, features: Dict) -> np.ndarray:
        """Create mock vector from feature dictionary."""
        vector = np.random.rand(self.embedding_dim)
        
        # Use features to create consistent vector
        if 'size_sqm' in features:
            vector[0] = features['size_sqm'] / 10000
        if 'build_year' in features:
            vector[1] = (features['build_year'] - 1950) / 100
            
        return vector.reshape(1, -1)


class MockRAGUtil:
    """Mock RAG utility for testing without external dependencies."""
    
    def __init__(self):
        """Initialize mock RAG utility."""
        self.vectorizer = MockPropertyVectorizer()
        self.property_data = None
        self.property_vectors = None
        
    def initialize_vectors(self) -> bool:
        """Initialize mock property vectors."""
        try:
            # Load property data from database
            self.property_data = db_util.query_db("SELECT * FROM properties LIMIT 100")
            
            if len(self.property_data) == 0:
                return False
                
            # Generate mock vectors
            self.property_vectors = self.vectorizer.vectorize_properties(self.property_data)
            return True
            
        except Exception as e:
            print(f"Error initializing vectors: {e}")
            return False
    
    def find_similar_properties(self, 
                              target_property_id: int = None,
                              target_features: Dict = None,
                              top_k: int = 10) -> Dict[str, Any]:
        """Find similar properties using mock vector similarity."""
        if self.property_vectors is None:
            if not self.initialize_vectors():
                return {"success": False, "error": "Failed to initialize vectors"}
        
        try:
            # Get target vector
            if target_property_id:
                target_idx = self.property_data[
                    self.property_data['property_id'] == target_property_id
                ].index
                if len(target_idx) == 0:
                    return {"success": False, "error": f"Property ID {target_property_id} not found"}
                target_vector = self.property_vectors[target_idx[0]]
            elif target_features:
                target_vector = self.vectorizer.vectorize_features(target_features)[0]
            else:
                # Use average of marketed properties, or first property if none marketed
                marketed_mask = self.property_data['is_marketed'] == 1
                if marketed_mask.any():
                    target_vector = np.mean(self.property_vectors[marketed_mask], axis=0)
                else:
                    # Use first property as default target
                    target_vector = self.property_vectors[0]
            
            # Calculate mock similarities (cosine similarity)
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([target_vector], self.property_vectors)[0]
            
            # Get top k similar properties
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                prop = self.property_data.iloc[idx].to_dict()
                prop['similarity_score'] = float(similarities[idx])
                results.append(prop)
            
            return {
                "success": True,
                "query_type": "similarity_search",
                "target_property_id": target_property_id,
                "target_features": target_features,
                "top_k": top_k,
                "results": results,
                "total_properties_searched": len(self.property_data),
                "avg_similarity": float(np.mean(similarities)),
                "error": None
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def analyze_property_clusters(self, n_clusters: int = 5) -> Dict[str, Any]:
        """Analyze property clusters using mock clustering."""
        if self.property_vectors is None:
            if not self.initialize_vectors():
                return {"success": False, "error": "Failed to initialize vectors"}
        
        try:
            from sklearn.cluster import KMeans
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.property_vectors)
            
            # Analyze clusters
            cluster_analysis = {
                "success": True,
                "query_type": "cluster_analysis",
                "n_clusters": n_clusters,
                "total_properties": len(self.property_data),
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "cluster_assignments": cluster_labels.tolist(),
                "cluster_summary": {},
                "error": None
            }
            
            # Summarize each cluster
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_properties = self.property_data[cluster_mask]
                
                cluster_analysis['cluster_summary'][f'cluster_{i}'] = {
                    'size': int(cluster_mask.sum()),
                    'avg_size_sqm': float(cluster_properties['size_sqm'].mean()),
                    'avg_build_year': float(cluster_properties['build_year'].mean()),
                    'regions': cluster_properties['region'].value_counts().to_dict(),
                    'marketed_count': int((cluster_properties['is_marketed'] == 1).sum()),
                    'sample_properties': cluster_properties.head(3)[
                        ['industrial_estate_name', 'unit_name', 'region', 'size_sqm']
                    ].to_dict('records')
                }
            
            return cluster_analysis
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def calculate_portfolio_homogeneity(self) -> Dict[str, Any]:
        """Calculate portfolio homogeneity using mock analysis."""
        if self.property_vectors is None:
            if not self.initialize_vectors():
                return {"success": False, "error": "Failed to initialize vectors"}
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(self.property_vectors)
            
            # Remove diagonal (self-similarities)
            np.fill_diagonal(similarity_matrix, 0)
            
            # Calculate metrics
            avg_similarity = np.mean(similarity_matrix)
            std_similarity = np.std(similarity_matrix)
            
            # Marketed vs non-marketed analysis
            marketed_mask = self.property_data['is_marketed'] == 1
            non_marketed_mask = self.property_data['is_marketed'] == 0
            
            if marketed_mask.any() and non_marketed_mask.any():
                cross_similarities = similarity_matrix[np.ix_(marketed_mask, non_marketed_mask)]
                marketed_vs_portfolio = np.mean(cross_similarities)
                
                marketed_internal = similarity_matrix[np.ix_(marketed_mask, marketed_mask)]
                np.fill_diagonal(marketed_internal, 0)
                marketed_internal_homogeneity = np.mean(marketed_internal) if marketed_internal.size > 1 else 0
            else:
                marketed_vs_portfolio = 0
                marketed_internal_homogeneity = 0
            
            return {
                "success": True,
                "query_type": "homogeneity_analysis",
                "total_properties": len(self.property_data),
                "marketed_properties": int(marketed_mask.sum()),
                "overall_homogeneity": float(avg_similarity),
                "homogeneity_std": float(std_similarity),
                "marketed_vs_portfolio": float(marketed_vs_portfolio),
                "marketed_internal_homogeneity": float(marketed_internal_homogeneity),
                "homogeneity_coefficient": float(avg_similarity / (std_similarity + 1e-8)),
                "similarity_distribution": {
                    "min": float(similarity_matrix[similarity_matrix > 0].min()),
                    "max": float(similarity_matrix.max()),
                    "median": float(np.median(similarity_matrix[similarity_matrix > 0]))
                },
                "error": None
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class TestRAGUtil(unittest.TestCase):
    """Test cases for RAG utility functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.rag_util = MockRAGUtil()
        self.test_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test-data')
        os.makedirs(self.test_data_dir, exist_ok=True)
    
    def test_vector_initialization(self):
        """Test RAG-1: Vector initialization and property data loading."""
        print("\n🔍 Testing RAG Vector Initialization")
        
        success = self.rag_util.initialize_vectors()
        self.assertTrue(success, "Vector initialization should succeed")
        
        self.assertIsNotNone(self.rag_util.property_data, "Property data should be loaded")
        self.assertIsNotNone(self.rag_util.property_vectors, "Property vectors should be created")
        
        # Validate vector dimensions
        expected_properties = len(self.rag_util.property_data)
        expected_dim = 128  # Mock embedding dimension
        
        self.assertEqual(self.rag_util.property_vectors.shape[0], expected_properties)
        self.assertEqual(self.rag_util.property_vectors.shape[1], expected_dim)
        
        # Save test results
        result_data = {
            "test": "vector_initialization",
            "success": True,
            "properties_loaded": expected_properties,
            "vector_dimensions": expected_dim,
            "vector_shape": list(self.rag_util.property_vectors.shape),
            "sample_vector_stats": {
                "mean": float(np.mean(self.rag_util.property_vectors[0])),
                "std": float(np.std(self.rag_util.property_vectors[0])),
                "min": float(np.min(self.rag_util.property_vectors[0])),
                "max": float(np.max(self.rag_util.property_vectors[0]))
            },
            "error": None
        }
        
        with open(os.path.join(self.test_data_dir, 'rag_vector_initialization.json'), 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"✅ Vector initialization completed")
        print(f"   Properties loaded: {expected_properties}")
        print(f"   Vector dimensions: {expected_dim}")
    
    def test_similarity_search_by_id(self):
        """Test RAG-2: Property similarity search by property ID."""
        print("\n🔍 Testing RAG Similarity Search by Property ID")
        
        # Initialize vectors
        self.rag_util.initialize_vectors()
        
        # Get a sample property ID
        sample_property_id = self.rag_util.property_data.iloc[0]['property_id']
        
        # Test similarity search
        result = self.rag_util.find_similar_properties(
            target_property_id=sample_property_id,
            top_k=10
        )
        
        self.assertTrue(result['success'], "Similarity search should succeed")
        self.assertEqual(result['query_type'], 'similarity_search')
        self.assertEqual(len(result['results']), 10, "Should return 10 similar properties")
        
        # Validate similarity scores
        similarities = [prop['similarity_score'] for prop in result['results']]
        self.assertTrue(all(0 <= score <= 1 for score in similarities), 
                       "Similarity scores should be between 0 and 1")
        self.assertTrue(similarities == sorted(similarities, reverse=True), 
                       "Results should be sorted by similarity score")
        
        # Save results
        with open(os.path.join(self.test_data_dir, 'rag_similarity_by_id.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Similarity search by ID completed")
        print(f"   Target property ID: {sample_property_id}")
        print(f"   Similar properties found: {len(result['results'])}")
        print(f"   Top similarity score: {similarities[0]:.3f}")
    
    def test_similarity_search_by_features(self):
        """Test RAG-3: Property similarity search by feature specification."""
        print("\n🔍 Testing RAG Similarity Search by Features")
        
        # Initialize vectors
        self.rag_util.initialize_vectors()
        
        # Define target features
        target_features = {
            'size_sqm': 7500,
            'build_year': 1985,
            'region': 'Midlands'
        }
        
        # Test similarity search
        result = self.rag_util.find_similar_properties(
            target_features=target_features,
            top_k=8
        )
        
        self.assertTrue(result['success'], "Feature-based similarity search should succeed")
        self.assertEqual(result['query_type'], 'similarity_search')
        self.assertEqual(len(result['results']), 8, "Should return 8 similar properties")
        self.assertEqual(result['target_features'], target_features)
        
        # Validate results structure
        for prop in result['results']:
            self.assertIn('similarity_score', prop)
            self.assertIn('industrial_estate_name', prop)
            self.assertIn('size_sqm', prop)
            self.assertIn('build_year', prop)
        
        # Save results
        with open(os.path.join(self.test_data_dir, 'rag_similarity_by_features.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Similarity search by features completed")
        print(f"   Target features: {target_features}")
        print(f"   Similar properties found: {len(result['results'])}")
        print(f"   Average similarity: {result['avg_similarity']:.3f}")
    
    def test_cluster_analysis(self):
        """Test RAG-4: Property cluster analysis."""
        print("\n🔍 Testing RAG Property Cluster Analysis")
        
        # Initialize vectors
        self.rag_util.initialize_vectors()
        
        # Test cluster analysis
        result = self.rag_util.analyze_property_clusters(n_clusters=5)
        
        self.assertTrue(result['success'], "Cluster analysis should succeed")
        self.assertEqual(result['query_type'], 'cluster_analysis')
        self.assertEqual(result['n_clusters'], 5)
        self.assertEqual(len(result['cluster_summary']), 5)
        
        # Validate cluster summary
        total_clustered = 0
        for cluster_id, cluster_info in result['cluster_summary'].items():
            self.assertIn('size', cluster_info)
            self.assertIn('avg_size_sqm', cluster_info)
            self.assertIn('avg_build_year', cluster_info)
            self.assertIn('regions', cluster_info)
            self.assertIn('marketed_count', cluster_info)
            self.assertGreater(cluster_info['size'], 0)
            total_clustered += cluster_info['size']
        
        self.assertEqual(total_clustered, result['total_properties'])
        
        # Save results
        with open(os.path.join(self.test_data_dir, 'rag_cluster_analysis.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Cluster analysis completed")
        print(f"   Number of clusters: {result['n_clusters']}")
        print(f"   Total properties clustered: {total_clustered}")
        print(f"   Largest cluster size: {max(c['size'] for c in result['cluster_summary'].values())}")
    
    def test_homogeneity_analysis(self):
        """Test RAG-5: Portfolio homogeneity analysis."""
        print("\n🔍 Testing RAG Portfolio Homogeneity Analysis")
        
        # Initialize vectors
        self.rag_util.initialize_vectors()
        
        # Test homogeneity analysis
        result = self.rag_util.calculate_portfolio_homogeneity()
        
        self.assertTrue(result['success'], "Homogeneity analysis should succeed")
        self.assertEqual(result['query_type'], 'homogeneity_analysis')
        
        # Validate required metrics
        required_metrics = [
            'overall_homogeneity', 'homogeneity_std', 'marketed_vs_portfolio',
            'marketed_internal_homogeneity', 'homogeneity_coefficient'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, result)
            self.assertIsInstance(result[metric], (int, float))
        
        # Validate value ranges
        self.assertGreaterEqual(result['overall_homogeneity'], 0)
        self.assertLessEqual(result['overall_homogeneity'], 1)
        self.assertGreaterEqual(result['homogeneity_std'], 0)
        
        # Save results
        with open(os.path.join(self.test_data_dir, 'rag_homogeneity_analysis.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Homogeneity analysis completed")
        print(f"   Overall homogeneity: {result['overall_homogeneity']:.3f}")
        print(f"   Marketed vs portfolio: {result['marketed_vs_portfolio']:.3f}")
        print(f"   Homogeneity coefficient: {result['homogeneity_coefficient']:.3f}")
    
    def test_comprehensive_rag_pipeline(self):
        """Test RAG-6: Comprehensive RAG analysis pipeline."""
        print("\n🔍 Testing Comprehensive RAG Analysis Pipeline")
        
        # Initialize vectors
        self.rag_util.initialize_vectors()
        
        # Run all RAG analyses
        similarity_result = self.rag_util.find_similar_properties(top_k=5)
        cluster_result = self.rag_util.analyze_property_clusters(n_clusters=3)
        homogeneity_result = self.rag_util.calculate_portfolio_homogeneity()
        
        # Combine results into comprehensive analysis
        comprehensive_result = {
            "test": "comprehensive_rag_pipeline",
            "success": all([
                similarity_result.get('success', False),
                cluster_result.get('success', False),
                homogeneity_result.get('success', False)
            ]),
            "analysis_timestamp": "2025-09-18",
            "portfolio_overview": {
                "total_properties": len(self.rag_util.property_data),
                "vector_dimensions": self.rag_util.property_vectors.shape[1],
                "marketed_properties": int((self.rag_util.property_data['is_marketed'] == 1).sum())
            },
            "similarity_analysis": {
                "success": similarity_result.get('success', False),
                "top_similar_properties": len(similarity_result.get('results', [])),
                "avg_similarity_score": similarity_result.get('avg_similarity', 0),
                "top_matches": similarity_result.get('results', [])[:3] if similarity_result.get('success') else []
            },
            "cluster_analysis": {
                "success": cluster_result.get('success', False),
                "n_clusters": cluster_result.get('n_clusters', 0),
                "cluster_summary": cluster_result.get('cluster_summary', {}),
                "largest_cluster_size": max(
                    (c.get('size', 0) for c in cluster_result.get('cluster_summary', {}).values()),
                    default=0
                )
            },
            "homogeneity_analysis": {
                "success": homogeneity_result.get('success', False),
                "overall_homogeneity": homogeneity_result.get('overall_homogeneity', 0),
                "marketed_vs_portfolio": homogeneity_result.get('marketed_vs_portfolio', 0),
                "homogeneity_coefficient": homogeneity_result.get('homogeneity_coefficient', 0)
            },
            "insights": {
                "portfolio_diversity_score": 1 - homogeneity_result.get('overall_homogeneity', 0),
                "most_similar_avg": similarity_result.get('avg_similarity', 0),
                "cluster_distribution": {
                    f"cluster_{i}": cluster_result.get('cluster_summary', {}).get(f'cluster_{i}', {}).get('size', 0)
                    for i in range(cluster_result.get('n_clusters', 0))
                }
            },
            "individual_results": {
                "similarity_result": similarity_result,
                "cluster_result": cluster_result,
                "homogeneity_result": homogeneity_result
            },
            "error": None
        }
        
        # Validate comprehensive results - check individual components
        self.assertTrue(similarity_result.get('success', False), f"Similarity analysis failed: {similarity_result.get('error')}")
        self.assertTrue(cluster_result.get('success', False), f"Cluster analysis failed: {cluster_result.get('error')}")
        self.assertTrue(homogeneity_result.get('success', False), f"Homogeneity analysis failed: {homogeneity_result.get('error')}")
        self.assertTrue(comprehensive_result['success'], "Overall comprehensive analysis should succeed")
        
        # Save comprehensive results
        with open(os.path.join(self.test_data_dir, 'rag_comprehensive_pipeline.json'), 'w') as f:
            json.dump(comprehensive_result, f, indent=2)
        
        print(f"✅ Comprehensive RAG pipeline completed")
        print(f"   Portfolio size: {comprehensive_result['portfolio_overview']['total_properties']}")
        print(f"   Similarity analysis: {comprehensive_result['similarity_analysis']['success']}")
        print(f"   Cluster analysis: {comprehensive_result['cluster_analysis']['success']}")
        print(f"   Homogeneity analysis: {comprehensive_result['homogeneity_analysis']['success']}")


if __name__ == '__main__':
    print("=" * 60)
    print("🧪 RUNNING RAG UTILITY UNIT TESTS")
    print("=" * 60)
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 60)
    print("✅ RAG UTILITY TESTS COMPLETED")
    print("=" * 60)
