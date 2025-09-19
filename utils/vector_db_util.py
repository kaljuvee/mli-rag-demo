"""
Vector database utility for MLI property data.
Handles vector embeddings and FAISS index operations.
"""

import os
import numpy as np
import pandas as pd
import pickle
import faiss
from typing import Dict, List, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

from .db_util import PropertyDatabase, db

class PropertyVectorDB:
    """Vector database for property data using FAISS."""
    
    def __init__(self, embeddings_dir=None):
        """Initialize vector database with embeddings directory."""
        if embeddings_dir is None:
            # Default path: embeddings/ relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.embeddings_dir = os.path.join(project_root, 'embeddings')
        else:
            self.embeddings_dir = embeddings_dir
        
        # Ensure directory exists
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Paths for saved files
        self.vectors_path = os.path.join(self.embeddings_dir, 'property_vectors.npy')
        self.faiss_index_path = os.path.join(self.embeddings_dir, 'faiss_index.bin')
        self.vectorizer_path = os.path.join(self.embeddings_dir, 'vectorizer.pkl')
        self.property_ids_path = os.path.join(self.embeddings_dir, 'property_ids.pkl')
        
        # Initialize components
        self.vectorizer = None
        self.property_vectors = None
        self.faiss_index = None
        self.property_ids = None
        self.property_data = None
        
        # Load if exists
        self._load_if_exists()
    
    def _load_if_exists(self):
        """Load existing vector database if available."""
        try:
            # Check if all required files exist
            if (os.path.exists(self.vectors_path) and 
                os.path.exists(self.faiss_index_path) and
                os.path.exists(self.vectorizer_path) and
                os.path.exists(self.property_ids_path)):
                
                # Load vectorizer
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                # Load property vectors
                self.property_vectors = np.load(self.vectors_path)
                
                # Load FAISS index
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                
                # Load property IDs
                with open(self.property_ids_path, 'rb') as f:
                    self.property_ids = pickle.load(f)
                
                print(f"✅ Loaded vector database from {self.embeddings_dir}")
                print(f"   - Vectors: {self.property_vectors.shape}")
                print(f"   - Properties: {len(self.property_ids)}")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Error loading vector database: {e}")
            # Reset components
            self.vectorizer = None
            self.property_vectors = None
            self.faiss_index = None
            self.property_ids = None
            return False
    
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
    
    def initialize_vectors(self, force_rebuild=False) -> bool:
        """Initialize property vectors from database."""
        # If already loaded and not forcing rebuild, return
        if self.property_vectors is not None and self.faiss_index is not None and not force_rebuild:
            return True
        
        try:
            print("🔄 Loading property data from database...")
            
            # Load all property data from SQL database
            sql_db = PropertyDatabase()
            self.property_data = sql_db.query_to_df("SELECT * FROM properties ORDER BY property_id")
            
            if len(self.property_data) == 0:
                raise ValueError("No property data found in database")
            
            print(f"📊 Loaded {len(self.property_data)} properties")
            print(f"   - Marketed properties: {(self.property_data['is_marketed'] == 1).sum()}")
            print(f"   - Portfolio properties: {(self.property_data['is_marketed'] == 0).sum()}")
            
            # Create text representations
            print("📝 Creating text representations...")
            property_texts = []
            for _, row in self.property_data.iterrows():
                text = self.create_property_text(row)
                property_texts.append(text)
            
            # Initialize vectorizer
            print("🤖 Initializing TF-IDF vectorizer...")
            self.vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
            
            # Generate embeddings
            print("🔢 Generating embeddings...")
            self.property_vectors = self.vectorizer.fit_transform(property_texts).toarray()
            
            # Store property IDs
            self.property_ids = self.property_data['property_id'].tolist()
            
            # Create FAISS index
            print("🔍 Creating FAISS index...")
            self.faiss_index = faiss.IndexFlatIP(self.property_vectors.shape[1])  # Inner product for cosine similarity
            
            # Normalize vectors for cosine similarity
            normalized_vectors = self.property_vectors / np.linalg.norm(self.property_vectors, axis=1, keepdims=True)
            self.faiss_index.add(normalized_vectors.astype('float32'))
            
            # Save to disk
            self._save_to_disk()
            
            print("✅ Vector database initialization completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing vector database: {e}")
            return False
    
    def _save_to_disk(self):
        """Save vector database to disk."""
        try:
            # Save vectorizer
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save property vectors
            np.save(self.vectors_path, self.property_vectors)
            
            # Save FAISS index
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            
            # Save property IDs
            with open(self.property_ids_path, 'wb') as f:
                pickle.dump(self.property_ids, f)
            
            print(f"✅ Saved vector database to {self.embeddings_dir}")
            
        except Exception as e:
            print(f"❌ Error saving vector database: {e}")
    
    def get_property_by_id(self, property_id):
        """Get property data by ID."""
        if self.property_data is None:
            # Load property data from SQL database
            sql_db = PropertyDatabase()
            self.property_data = sql_db.query_to_df("SELECT * FROM properties ORDER BY property_id")
        
        return self.property_data[self.property_data['property_id'] == property_id]
    
    def get_all_properties(self):
        """Get all property data."""
        if self.property_data is None:
            # Load property data from SQL database
            sql_db = PropertyDatabase()
            self.property_data = sql_db.query_to_df("SELECT * FROM properties ORDER BY property_id")
        
        return self.property_data
    
    def get_marketed_properties(self):
        """Get marketed properties."""
        all_properties = self.get_all_properties()
        return all_properties[all_properties['is_marketed'] == 1]
    
    def vectorize_query(self, query_text: str) -> np.ndarray:
        """Create embedding for a query text."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not initialized")
        
        # Transform query text to vector
        query_vector = self.vectorizer.transform([query_text]).toarray()[0]
        
        # Normalize vector
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        return query_vector
    
    def find_similar_properties(self, 
                              target_property_id: Optional[int] = None,
                              query_text: Optional[str] = None,
                              top_k: int = 10) -> Dict[str, Any]:
        """Find similar properties using vector similarity."""
        if not self.initialize_vectors():
            return {"success": False, "error": "Failed to initialize vectors"}
        
        try:
            # Get target vector
            if target_property_id is not None:
                # Find property by ID
                property_idx = None
                for i, pid in enumerate(self.property_ids):
                    if pid == target_property_id:
                        property_idx = i
                        break
                
                if property_idx is None:
                    return {"success": False, "error": f"Property ID {target_property_id} not found"}
                
                target_vector = self.property_vectors[property_idx]
                query_description = f"Property ID {target_property_id}"
                
            elif query_text:
                # Create embedding for query text
                target_vector = self.vectorize_query(query_text)
                query_description = query_text
                
            else:
                # Use average of marketed properties as default
                all_properties = self.get_all_properties()
                marketed_indices = []
                for i, pid in enumerate(self.property_ids):
                    if pid in all_properties[all_properties['is_marketed'] == 1]['property_id'].values:
                        marketed_indices.append(i)
                
                if not marketed_indices:
                    return {"success": False, "error": "No marketed properties found and no query specified"}
                
                target_vector = np.mean(self.property_vectors[marketed_indices], axis=0)
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
                property_id = self.property_ids[idx]
                property_data = self.get_property_by_id(property_id)
                
                if len(property_data) > 0:
                    prop = property_data.iloc[0].to_dict()
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
                "total_properties_searched": len(self.property_ids),
                "avg_similarity": float(np.mean(similarities[0])),
                "max_similarity": float(np.max(similarities[0])),
                "min_similarity": float(np.min(similarities[0])),
                "embedding_model": "TF-IDF",
                "error": None
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def calculate_portfolio_homogeneity(self) -> Dict[str, Any]:
        """Calculate portfolio homogeneity using embeddings."""
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
            
            # Get property data
            all_properties = self.get_all_properties()
            
            # Create masks for marketed and non-marketed properties
            marketed_indices = []
            non_marketed_indices = []
            
            for i, pid in enumerate(self.property_ids):
                if pid in all_properties[all_properties['is_marketed'] == 1]['property_id'].values:
                    marketed_indices.append(i)
                else:
                    non_marketed_indices.append(i)
            
            # Marketed vs non-marketed analysis
            if marketed_indices and non_marketed_indices:
                # Cross-similarity between marketed and non-marketed
                cross_similarities = similarity_matrix[np.ix_(marketed_indices, non_marketed_indices)]
                marketed_vs_portfolio = np.mean(cross_similarities)
                
                # Internal similarities within marketed properties
                if len(marketed_indices) > 1:
                    marketed_internal = similarity_matrix[np.ix_(marketed_indices, marketed_indices)]
                    np.fill_diagonal(marketed_internal, 0)
                    marketed_internal_homogeneity = np.mean(marketed_internal)
                else:
                    marketed_internal_homogeneity = 0
                
                # Internal similarities within non-marketed properties
                non_marketed_internal = similarity_matrix[np.ix_(non_marketed_indices, non_marketed_indices)]
                np.fill_diagonal(non_marketed_internal, 0)
                portfolio_internal_homogeneity = np.mean(non_marketed_internal)
                
            else:
                marketed_vs_portfolio = 0
                marketed_internal_homogeneity = 0
                portfolio_internal_homogeneity = avg_similarity
            
            # Calculate homogeneity coefficient
            homogeneity_coefficient = avg_similarity / (std_similarity + 1e-8)
            
            # Calculate similarity distribution statistics
            flat_similarities = similarity_matrix[similarity_matrix > 0].flatten()
            similarity_distribution = {
                "min": float(np.min(flat_similarities)),
                "max": float(np.max(flat_similarities)),
                "median": float(np.median(flat_similarities)),
                "percentile_25": float(np.percentile(flat_similarities, 25)),
                "percentile_75": float(np.percentile(flat_similarities, 75))
            }
            
            result = {
                "success": True,
                "query_type": "homogeneity_analysis",
                "total_properties": len(self.property_ids),
                "marketed_properties": len(marketed_indices),
                "portfolio_properties": len(non_marketed_indices),
                "embedding_model": "TF-IDF",
                "overall_homogeneity": float(avg_similarity),
                "homogeneity_std": float(std_similarity),
                "homogeneity_coefficient": float(homogeneity_coefficient),
                "marketed_vs_portfolio": float(marketed_vs_portfolio),
                "marketed_internal_homogeneity": float(marketed_internal_homogeneity),
                "portfolio_internal_homogeneity": float(portfolio_internal_homogeneity),
                "similarity_distribution": similarity_distribution,
                "error": None
            }
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# Create a global instance for convenience
vector_db = PropertyVectorDB()
