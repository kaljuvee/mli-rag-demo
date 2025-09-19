"""
RAG Chat utility for property analysis using vector embeddings and similarity search.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle

from . import db_util
from .rag_util import PropertyVectorizer


class RAGChat:
    """RAG-based chat system for property analysis using vector embeddings."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize RAG chat system."""
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.vectorizer = PropertyVectorizer(openai_api_key=self.api_key)
        self.property_vectors = None
        self.property_data = None
        self.scaler = StandardScaler()
        
    def initialize_vectors(self) -> bool:
        """Initialize property vectors from database."""
        try:
            # Load property data
            self.property_data = db_util.query_db("""
                SELECT * FROM properties 
                WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            """)
            
            if len(self.property_data) == 0:
                raise ValueError("No property data found")
            
            # Generate vectors for all properties
            self.property_vectors = self.vectorizer.vectorize_properties(self.property_data)
            
            return True
            
        except Exception as e:
            print(f"Error initializing vectors: {e}")
            return False
    
    def find_similar_properties(self, 
                              target_property_id: Optional[int] = None,
                              target_features: Optional[Dict] = None,
                              top_k: int = 10,
                              exclude_marketed: bool = True) -> pd.DataFrame:
        """
        Find similar properties using vector similarity.
        
        Args:
            target_property_id: ID of target property to find similarities for
            target_features: Dictionary of features to match against
            top_k: Number of similar properties to return
            exclude_marketed: Whether to exclude marketed properties
            
        Returns:
            DataFrame with similar properties and similarity scores
        """
        if self.property_vectors is None:
            if not self.initialize_vectors():
                raise RuntimeError("Failed to initialize property vectors")
        
        # Get target vector
        if target_property_id:
            target_idx = self.property_data[
                self.property_data['property_id'] == target_property_id
            ].index
            if len(target_idx) == 0:
                raise ValueError(f"Property ID {target_property_id} not found")
            target_vector = self.property_vectors[target_idx[0]].reshape(1, -1)
            
        elif target_features:
            # Create synthetic property from features
            target_vector = self.vectorizer.vectorize_features(target_features)
            
        else:
            # Use average of marketed properties as target
            marketed_mask = self.property_data['is_marketed'] == 1
            if not marketed_mask.any():
                raise ValueError("No marketed properties found")
            target_vector = np.mean(self.property_vectors[marketed_mask], axis=0).reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(target_vector, self.property_vectors)[0]
        
        # Create results dataframe
        results_df = self.property_data.copy()
        results_df['similarity_score'] = similarities
        
        # Apply filters
        if exclude_marketed:
            results_df = results_df[results_df['is_marketed'] == 0]
        
        if target_property_id:
            results_df = results_df[results_df['property_id'] != target_property_id]
        
        # Sort by similarity and return top k
        results_df = results_df.sort_values('similarity_score', ascending=False)
        return results_df.head(top_k)
    
    def analyze_property_clusters(self, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Analyze property clusters using vector embeddings.
        
        Args:
            n_clusters: Number of clusters to identify
            
        Returns:
            Dictionary with cluster analysis results
        """
        if self.property_vectors is None:
            if not self.initialize_vectors():
                raise RuntimeError("Failed to initialize property vectors")
        
        from sklearn.cluster import KMeans
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.property_vectors)
        
        # Analyze clusters
        results = {
            'n_clusters': n_clusters,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_assignments': cluster_labels.tolist(),
            'cluster_summary': {}
        }
        
        # Summarize each cluster
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_properties = self.property_data[cluster_mask]
            
            results['cluster_summary'][f'cluster_{i}'] = {
                'size': int(cluster_mask.sum()),
                'avg_size_sqm': float(cluster_properties['size_sqm'].mean()),
                'avg_build_year': float(cluster_properties['build_year'].mean()),
                'regions': cluster_properties['region'].value_counts().to_dict(),
                'marketed_count': int((cluster_properties['is_marketed'] == 1).sum())
            }
        
        return results
    
    def calculate_portfolio_homogeneity(self) -> Dict[str, float]:
        """
        Calculate homogeneity scores for the property portfolio.
        
        Returns:
            Dictionary with various homogeneity metrics
        """
        if self.property_vectors is None:
            if not self.initialize_vectors():
                raise RuntimeError("Failed to initialize property vectors")
        
        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(self.property_vectors)
        
        # Remove diagonal (self-similarities)
        np.fill_diagonal(similarity_matrix, 0)
        
        # Calculate metrics
        avg_similarity = np.mean(similarity_matrix)
        std_similarity = np.std(similarity_matrix)
        
        # Marketed vs non-marketed homogeneity
        marketed_mask = self.property_data['is_marketed'] == 1
        non_marketed_mask = self.property_data['is_marketed'] == 0
        
        if marketed_mask.any() and non_marketed_mask.any():
            # Cross-similarity between marketed and non-marketed
            cross_similarities = similarity_matrix[np.ix_(marketed_mask, non_marketed_mask)]
            marketed_nonmarketed_similarity = np.mean(cross_similarities)
            
            # Internal similarities
            marketed_internal = similarity_matrix[np.ix_(marketed_mask, marketed_mask)]
            np.fill_diagonal(marketed_internal, 0)
            marketed_internal_similarity = np.mean(marketed_internal) if marketed_internal.size > 1 else 0
            
        else:
            marketed_nonmarketed_similarity = 0
            marketed_internal_similarity = 0
        
        return {
            'overall_homogeneity': float(avg_similarity),
            'homogeneity_std': float(std_similarity),
            'marketed_vs_portfolio': float(marketed_nonmarketed_similarity),
            'marketed_internal_homogeneity': float(marketed_internal_similarity),
            'homogeneity_coefficient': float(avg_similarity / (std_similarity + 1e-8))
        }
    
    def query_with_context(self, query: str) -> Dict[str, Any]:
        """
        Answer queries using RAG with property context.
        
        Args:
            query: Natural language query about properties
            
        Returns:
            Dictionary with query results and context
        """
        try:
            # Determine query type and extract relevant properties
            if "similar" in query.lower():
                # Find similar properties
                similar_props = self.find_similar_properties(top_k=10)
                context_data = similar_props.head(5)
                analysis_type = "similarity_search"
                
            elif "cluster" in query.lower() or "group" in query.lower():
                # Cluster analysis
                cluster_results = self.analyze_property_clusters()
                context_data = cluster_results
                analysis_type = "cluster_analysis"
                
            elif "homogen" in query.lower() or "correlation" in query.lower():
                # Homogeneity analysis
                homogeneity_results = self.calculate_portfolio_homogeneity()
                context_data = homogeneity_results
                analysis_type = "homogeneity_analysis"
                
            else:
                # General property search
                context_data = self.property_data.head(10)
                analysis_type = "general_search"
            
            # Generate response using OpenAI
            context_str = self._format_context_for_llm(context_data, analysis_type)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a property analysis expert. Use the provided property data and analysis results to answer questions about real estate portfolios. Focus on insights about property similarities, market trends, and portfolio composition."""
                    },
                    {
                        "role": "user", 
                        "content": f"Query: {query}\n\nProperty Analysis Context:\n{context_str}\n\nPlease provide a detailed analysis based on this data."
                    }
                ],
                temperature=0.1
            )
            
            return {
                "query": query,
                "success": True,
                "analysis_type": analysis_type,
                "response": response.choices[0].message.content,
                "context_data": context_data.to_dict('records') if hasattr(context_data, 'to_dict') else context_data,
                "error": None
            }
            
        except Exception as e:
            return {
                "query": query,
                "success": False,
                "analysis_type": "error",
                "response": f"Error processing query: {str(e)}",
                "context_data": None,
                "error": str(e)
            }
    
    def _format_context_for_llm(self, context_data: Any, analysis_type: str) -> str:
        """Format context data for LLM consumption."""
        if analysis_type == "similarity_search" and hasattr(context_data, 'to_string'):
            return f"Similar Properties Analysis:\n{context_data[['industrial_estate_name', 'unit_name', 'region', 'size_sqm', 'build_year', 'similarity_score']].to_string()}"
            
        elif analysis_type == "cluster_analysis":
            return f"Property Cluster Analysis:\n{json.dumps(context_data, indent=2)}"
            
        elif analysis_type == "homogeneity_analysis":
            return f"Portfolio Homogeneity Analysis:\n{json.dumps(context_data, indent=2)}"
            
        elif hasattr(context_data, 'to_string'):
            return f"Property Data Sample:\n{context_data[['industrial_estate_name', 'unit_name', 'region', 'size_sqm', 'build_year']].to_string()}"
            
        else:
            return str(context_data)


class MockRAGChat:
    """Mock RAG chat for testing without OpenAI API calls."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize mock RAG chat."""
        self.vectorizer = None
        self.property_vectors = None
        self.property_data = None
        
    def initialize_vectors(self) -> bool:
        """Mock vector initialization."""
        try:
            self.property_data = db_util.query_db("SELECT * FROM properties LIMIT 100")
            # Create mock vectors
            n_properties = len(self.property_data)
            self.property_vectors = np.random.rand(n_properties, 10)  # 10-dim mock vectors
            return True
        except:
            return False
    
    def find_similar_properties(self, target_property_id=None, target_features=None, top_k=10, exclude_marketed=True):
        """Mock similarity search."""
        if self.property_data is None:
            self.initialize_vectors()
        
        # Return mock similar properties
        result_df = self.property_data.head(top_k).copy()
        result_df['similarity_score'] = np.random.uniform(0.7, 0.95, len(result_df))
        return result_df.sort_values('similarity_score', ascending=False)
    
    def analyze_property_clusters(self, n_clusters=5):
        """Mock cluster analysis."""
        return {
            'n_clusters': n_clusters,
            'cluster_summary': {
                f'cluster_{i}': {
                    'size': 20 + i * 5,
                    'avg_size_sqm': 5000 + i * 1000,
                    'avg_build_year': 1980 + i * 5,
                    'regions': {'Midlands': 10, 'London': 8, 'North West': 5},
                    'marketed_count': 1 if i == 0 else 0
                }
                for i in range(n_clusters)
            }
        }
    
    def calculate_portfolio_homogeneity(self):
        """Mock homogeneity calculation."""
        return {
            'overall_homogeneity': 0.72,
            'homogeneity_std': 0.15,
            'marketed_vs_portfolio': 0.68,
            'marketed_internal_homogeneity': 0.85,
            'homogeneity_coefficient': 4.8
        }
    
    def query_with_context(self, query: str):
        """Mock query processing."""
        if "similar" in query.lower():
            similar_props = self.find_similar_properties(top_k=5)
            return {
                "query": query,
                "success": True,
                "analysis_type": "similarity_search",
                "response": f"Found {len(similar_props)} similar properties with similarity scores ranging from {similar_props['similarity_score'].min():.3f} to {similar_props['similarity_score'].max():.3f}. The most similar property is {similar_props.iloc[0]['industrial_estate_name']} {similar_props.iloc[0]['unit_name']} with a similarity score of {similar_props.iloc[0]['similarity_score']:.3f}.",
                "context_data": similar_props.to_dict('records'),
                "error": None
            }
        
        elif "cluster" in query.lower():
            clusters = self.analyze_property_clusters()
            return {
                "query": query,
                "success": True,
                "analysis_type": "cluster_analysis", 
                "response": f"Portfolio analysis reveals {clusters['n_clusters']} distinct property clusters. The largest cluster contains {max([c['size'] for c in clusters['cluster_summary'].values()])} properties, with average sizes ranging from {min([c['avg_size_sqm'] for c in clusters['cluster_summary'].values()])} to {max([c['avg_size_sqm'] for c in clusters['cluster_summary'].values()])} sqm.",
                "context_data": clusters,
                "error": None
            }
        
        elif "homogen" in query.lower():
            homogeneity = self.calculate_portfolio_homogeneity()
            return {
                "query": query,
                "success": True,
                "analysis_type": "homogeneity_analysis",
                "response": f"Portfolio homogeneity analysis shows an overall homogeneity score of {homogeneity['overall_homogeneity']:.3f}. The marketed properties have a {homogeneity['marketed_vs_portfolio']:.3f} similarity to the rest of the portfolio, indicating {'good' if homogeneity['marketed_vs_portfolio'] > 0.6 else 'moderate'} alignment with the existing portfolio characteristics.",
                "context_data": homogeneity,
                "error": None
            }
        
        else:
            return {
                "query": query,
                "success": True,
                "analysis_type": "general_search",
                "response": f"General property analysis for query: '{query}'. The portfolio contains {len(self.property_data) if self.property_data is not None else 1255} properties across multiple regions with diverse characteristics.",
                "context_data": {"total_properties": len(self.property_data) if self.property_data is not None else 1255},
                "error": None
            }
