"""
Real RAG utility for property analysis using actual OpenAI embeddings and FAISS.
No mock data - uses real vector database and embeddings.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from openai import OpenAI
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

from . import db_util


class RealPropertyVectorizer:
    """Real property vectorizer using OpenAI embeddings."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize with OpenAI client."""
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.embedding_model = "text-embedding-ada-002"
        self.embedding_dim = 1536  # Dimension for text-embedding-ada-002
        
    def create_property_text(self, property_row: pd.Series) -> str:
        """Create text representation of property for embedding."""
        text_parts = []
        
        # Add estate and unit information
        if pd.notna(property_row.get('industrial_estate_name')):
            text_parts.append(f"Industrial Estate: {property_row['industrial_estate_name']}")
        if pd.notna(property_row.get('unit_name')):
            text_parts.append(f"Unit: {property_row['unit_name']}")
        
        # Add location information
        if pd.notna(property_row.get('region')):
            text_parts.append(f"Region: {property_row['region']}")
        
        # Add size and physical characteristics
        if pd.notna(property_row.get('size_sqm')):
            text_parts.append(f"Size: {property_row['size_sqm']:.0f} square meters")
        if pd.notna(property_row.get('build_year')):
            text_parts.append(f"Built in: {property_row['build_year']:.0f}")
        
        # Add physical features
        if pd.notna(property_row.get('car_parking_spaces')) and property_row['car_parking_spaces'] > 0:
            text_parts.append(f"Parking spaces: {property_row['car_parking_spaces']}")
        if pd.notna(property_row.get('min_eaves_m')) and property_row['min_eaves_m'] > 0:
            text_parts.append(f"Minimum eaves height: {property_row['min_eaves_m']:.1f} meters")
        if pd.notna(property_row.get('max_eaves_m')) and property_row['max_eaves_m'] > 0:
            text_parts.append(f"Maximum eaves height: {property_row['max_eaves_m']:.1f} meters")
        if pd.notna(property_row.get('doors')) and property_row['doors'] > 0:
            text_parts.append(f"Loading doors: {property_row['doors']}")
        
        # Add EPC rating
        if pd.notna(property_row.get('epc_rating')) and property_row['epc_rating'] != 'Not Available':
            text_parts.append(f"EPC Rating: {property_row['epc_rating']}")
        
        # Add marketing status
        if property_row.get('is_marketed', 0) == 1:
            text_parts.append("Currently marketed property")
        else:
            text_parts.append("Portfolio property")
        
        return ". ".join(text_parts) + "."
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API."""
        try:
            # Process in batches to avoid API limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return np.array(all_embeddings)
            
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            raise
    
    def vectorize_properties(self, properties_df: pd.DataFrame) -> np.ndarray:
        """Create embeddings for all properties."""
        print(f"Creating embeddings for {len(properties_df)} properties...")
        
        # Create text representations
        property_texts = []
        for _, row in properties_df.iterrows():
            text = self.create_property_text(row)
            property_texts.append(text)
        
        # Get embeddings
        embeddings = self.get_embeddings(property_texts)
        
        print(f"✅ Created embeddings: {embeddings.shape}")
        return embeddings
    
    def vectorize_query(self, query_text: str) -> np.ndarray:
        """Create embedding for a query text."""
        embeddings = self.get_embeddings([query_text])
        return embeddings[0]


class RealRAGUtil:
    """Real RAG utility using actual OpenAI embeddings and FAISS."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize RAG utility."""
        self.vectorizer = RealPropertyVectorizer(openai_api_key)
        self.property_data = None
        self.property_vectors = None
        self.faiss_index = None
        
    def initialize_vectors(self) -> bool:
        """Initialize property vectors from database."""
        try:
            print("🔄 Loading property data from database...")
            
            # Load all property data
            self.property_data = db_util.query_db("""
                SELECT * FROM properties 
                ORDER BY property_id
            """)
            
            if len(self.property_data) == 0:
                raise ValueError("No property data found in database")
            
            print(f"📊 Loaded {len(self.property_data)} properties")
            print(f"   - Marketed properties: {(self.property_data['is_marketed'] == 1).sum()}")
            print(f"   - Portfolio properties: {(self.property_data['is_marketed'] == 0).sum()}")
            
            # Generate embeddings
            print("🤖 Generating OpenAI embeddings...")
            self.property_vectors = self.vectorizer.vectorize_properties(self.property_data)
            
            # Create FAISS index
            print("🔍 Creating FAISS index...")
            self.faiss_index = faiss.IndexFlatIP(self.property_vectors.shape[1])  # Inner product for cosine similarity
            
            # Normalize vectors for cosine similarity
            normalized_vectors = self.property_vectors / np.linalg.norm(self.property_vectors, axis=1, keepdims=True)
            self.faiss_index.add(normalized_vectors.astype('float32'))
            
            print("✅ Vector initialization completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing vectors: {e}")
            return False
    
    def find_similar_properties(self, 
                              target_property_id: Optional[int] = None,
                              query_text: Optional[str] = None,
                              top_k: int = 10) -> Dict[str, Any]:
        """Find similar properties using real vector similarity."""
        if self.property_vectors is None or self.faiss_index is None:
            if not self.initialize_vectors():
                return {"success": False, "error": "Failed to initialize vectors"}
        
        try:
            # Get target vector
            if target_property_id:
                # Find property by ID
                target_idx = self.property_data[
                    self.property_data['property_id'] == target_property_id
                ].index
                if len(target_idx) == 0:
                    return {"success": False, "error": f"Property ID {target_property_id} not found"}
                
                target_vector = self.property_vectors[target_idx[0]]
                query_description = f"Property ID {target_property_id}"
                
            elif query_text:
                # Create embedding for query text
                target_vector = self.vectorizer.vectorize_query(query_text)
                query_description = query_text
                
            else:
                # Use average of marketed properties as default
                marketed_mask = self.property_data['is_marketed'] == 1
                if not marketed_mask.any():
                    return {"success": False, "error": "No marketed properties found and no query specified"}
                
                target_vector = np.mean(self.property_vectors[marketed_mask], axis=0)
                query_description = "Average of marketed properties"
            
            # Normalize target vector
            target_vector = target_vector / np.linalg.norm(target_vector)
            
            # Search using FAISS
            similarities, indices = self.faiss_index.search(
                target_vector.reshape(1, -1).astype('float32'), 
                top_k
            )
            
            # Prepare results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                prop = self.property_data.iloc[idx].to_dict()
                prop['similarity_score'] = float(similarity)
                prop['rank'] = i + 1
                results.append(prop)
            
            return {
                "success": True,
                "query_type": "similarity_search",
                "query_description": query_description,
                "target_property_id": target_property_id,
                "query_text": query_text,
                "top_k": top_k,
                "results": results,
                "total_properties_searched": len(self.property_data),
                "avg_similarity": float(np.mean(similarities[0])),
                "max_similarity": float(np.max(similarities[0])),
                "min_similarity": float(np.min(similarities[0])),
                "error": None
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def analyze_property_clusters(self, n_clusters: int = 5) -> Dict[str, Any]:
        """Analyze property clusters using real embeddings."""
        if self.property_vectors is None:
            if not self.initialize_vectors():
                return {"success": False, "error": "Failed to initialize vectors"}
        
        try:
            print(f"🔍 Performing K-means clustering with {n_clusters} clusters...")
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.property_vectors)
            
            # Analyze clusters
            cluster_analysis = {
                "success": True,
                "query_type": "cluster_analysis",
                "n_clusters": n_clusters,
                "total_properties": len(self.property_data),
                "cluster_centers_shape": list(kmeans.cluster_centers_.shape),
                "cluster_assignments": cluster_labels.tolist(),
                "cluster_summary": {},
                "error": None
            }
            
            # Summarize each cluster
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_properties = self.property_data[cluster_mask]
                
                # Calculate cluster statistics
                cluster_info = {
                    'size': int(cluster_mask.sum()),
                    'percentage': float(cluster_mask.sum() / len(self.property_data) * 100),
                    'avg_size_sqm': float(cluster_properties['size_sqm'].mean()),
                    'avg_build_year': float(cluster_properties['build_year'].mean()),
                    'regions': cluster_properties['region'].value_counts().to_dict(),
                    'marketed_count': int((cluster_properties['is_marketed'] == 1).sum()),
                    'avg_parking_spaces': float(cluster_properties['car_parking_spaces'].mean()),
                    'epc_ratings': cluster_properties['epc_rating'].value_counts().to_dict(),
                    'sample_properties': cluster_properties.head(3)[
                        ['industrial_estate_name', 'unit_name', 'region', 'size_sqm', 'build_year']
                    ].to_dict('records')
                }
                
                cluster_analysis['cluster_summary'][f'cluster_{i}'] = cluster_info
            
            print("✅ Cluster analysis completed")
            return cluster_analysis
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def calculate_portfolio_homogeneity(self) -> Dict[str, Any]:
        """Calculate portfolio homogeneity using real embeddings."""
        if self.property_vectors is None:
            if not self.initialize_vectors():
                return {"success": False, "error": "Failed to initialize vectors"}
        
        try:
            print("📊 Calculating portfolio homogeneity...")
            
            # Calculate pairwise similarities using normalized vectors
            normalized_vectors = self.property_vectors / np.linalg.norm(self.property_vectors, axis=1, keepdims=True)
            similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
            
            # Remove diagonal (self-similarities)
            np.fill_diagonal(similarity_matrix, 0)
            
            # Calculate overall metrics
            avg_similarity = np.mean(similarity_matrix)
            std_similarity = np.std(similarity_matrix)
            
            # Marketed vs non-marketed analysis
            marketed_mask = self.property_data['is_marketed'] == 1
            non_marketed_mask = self.property_data['is_marketed'] == 0
            
            if marketed_mask.any() and non_marketed_mask.any():
                # Cross-similarity between marketed and non-marketed
                cross_similarities = similarity_matrix[np.ix_(marketed_mask, non_marketed_mask)]
                marketed_vs_portfolio = np.mean(cross_similarities)
                
                # Internal similarities within marketed properties
                marketed_internal = similarity_matrix[np.ix_(marketed_mask, marketed_mask)]
                np.fill_diagonal(marketed_internal, 0)
                marketed_internal_homogeneity = np.mean(marketed_internal) if marketed_internal.size > 1 else 0
                
                # Internal similarities within non-marketed properties
                non_marketed_internal = similarity_matrix[np.ix_(non_marketed_mask, non_marketed_mask)]
                np.fill_diagonal(non_marketed_internal, 0)
                portfolio_internal_homogeneity = np.mean(non_marketed_internal)
                
            else:
                marketed_vs_portfolio = 0
                marketed_internal_homogeneity = 0
                portfolio_internal_homogeneity = avg_similarity
            
            # Calculate homogeneity coefficient
            homogeneity_coefficient = avg_similarity / (std_similarity + 1e-8)
            
            result = {
                "success": True,
                "query_type": "homogeneity_analysis",
                "total_properties": len(self.property_data),
                "marketed_properties": int(marketed_mask.sum()),
                "portfolio_properties": int(non_marketed_mask.sum()),
                "overall_homogeneity": float(avg_similarity),
                "homogeneity_std": float(std_similarity),
                "marketed_vs_portfolio": float(marketed_vs_portfolio),
                "marketed_internal_homogeneity": float(marketed_internal_homogeneity),
                "portfolio_internal_homogeneity": float(portfolio_internal_homogeneity),
                "homogeneity_coefficient": float(homogeneity_coefficient),
                "similarity_distribution": {
                    "min": float(similarity_matrix[similarity_matrix > 0].min()) if (similarity_matrix > 0).any() else 0,
                    "max": float(similarity_matrix.max()),
                    "median": float(np.median(similarity_matrix[similarity_matrix > 0])) if (similarity_matrix > 0).any() else 0,
                    "percentile_25": float(np.percentile(similarity_matrix[similarity_matrix > 0], 25)) if (similarity_matrix > 0).any() else 0,
                    "percentile_75": float(np.percentile(similarity_matrix[similarity_matrix > 0], 75)) if (similarity_matrix > 0).any() else 0
                },
                "error": None
            }
            
            print("✅ Homogeneity analysis completed")
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}


def create_real_rag_util(openai_api_key: Optional[str] = None) -> RealRAGUtil:
    """Factory function to create RealRAGUtil instance."""
    return RealRAGUtil(openai_api_key=openai_api_key)
