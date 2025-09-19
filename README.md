# MLI RAG Demo - AI-Powered Property Portfolio Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32+-red.svg)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/openai-gpt--4o--mini-green.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive **Retrieval-Augmented Generation (RAG)** demonstration application for real estate portfolio analysis, featuring advanced AI-powered querying, property similarity analysis, and intelligent data insights. Built for the MLI (Machine Learning Infrastructure) assessment, this application showcases modern AI engineering practices with production-ready architecture.

## 🚀 Live Demo

The application is deployed and available at:

- **Streamlit Web Interface**: [https://mli-rag-demo.streamlit.app](https://mli-rag-demo.streamlit.app)
- **API Documentation**: 
  - Swagger UI: [https://mli-rag-demo-api.onrender.com/docs](https://mli-rag-demo-api.onrender.com/docs)
  - OpenAPI JSON: [https://mli-rag-demo-api.onrender.com/openapi.json](https://mli-rag-demo-api.onrender.com/openapi.json)

## 🏗️ Architecture Overview

The MLI RAG Demo implements a sophisticated multi-layered architecture combining:

- **🤖 AI-Powered Text-to-SQL**: Natural language queries converted to SQL using GPT-4o-mini
- **🔍 Vector-Based RAG Analysis**: Property similarity search using TF-IDF embeddings and FAISS
- **📊 Interactive Web Interface**: Multi-page Streamlit application with real-time processing
- **🗄️ Robust Data Pipeline**: Automated Excel processing with SQLite storage
- **🧪 Comprehensive Testing**: Unit tests with JSON result validation
- **🚀 Production-Ready Deployment**: FastAPI backend with RESTful endpoints

## 📁 Project Directory Structure

```
mli-rag-demo/
├── 📁 api/                          # API Documentation
│   └── swagger.json                 # OpenAPI specification
│
├── 📁 data/                         # Raw Data Files
│   ├── CurrentPortfolio.xlsx        # 1,250 current properties
│   └── MarketedWarehouses.xlsx      # 5 marketed properties
│
├── 📁 db/                           # SQL Database Storage
│   └── mli.db                       # SQLite database (auto-generated)
│
├── 📁 embeddings/                   # Vector Database Storage
│   ├── property_vectors.npy         # Property embeddings array
│   ├── faiss_index.bin              # FAISS similarity index
│   ├── vectorizer.pkl               # TF-IDF vectorizer model
│   └── property_ids.pkl             # Property ID mapping
│
├── 📁 pages/                        # Streamlit Multi-Page Application
│   ├── 1_Preprocess.py              # Data loading and preprocessing
│   ├── 2_Text-to-SQL.py             # Natural language SQL queries
│   ├── 3_RAG_Analysis.py            # Vector-based property analysis
│   └── 4_Simple_Query.py            # Basic database queries (testing)
│
├── 📁 test-data/                    # Unit Test Results (JSON)
│   ├── README.md                    # Test data documentation
│   ├── FINAL_TEST_RESULTS.md        # Comprehensive test summary
│   ├── TEST_RESULTS_SUMMARY.md      # Detailed test analysis
│   ├── query_1_similar_properties.json      # Similarity analysis results
│   ├── query_2_correlation_score.json       # Correlation analysis results
│   ├── query_3_closest_properties.json      # Geographic proximity results
│   ├── simple_queries.json                  # Basic functionality tests
│   ├── sql_generation_results.json          # SQL validation tests
│   ├── rag_similarity_search.json           # RAG similarity tests
│   ├── rag_cluster_analysis.json            # RAG clustering tests
│   ├── rag_homogeneity_analysis.json        # RAG homogeneity tests
│   └── rag_comprehensive_analysis.json      # Complete RAG analysis
│
├── 📁 tests/                        # Unit Test Suite
│   ├── test_sql_chat.py             # SQL chat functionality tests
│   ├── test_rag_util.py             # RAG functionality tests
│   ├── test_hybrid_rag_simple.py    # Hybrid RAG implementation tests
│   ├── test_xls_converter.py        # Data processing tests
│   └── test_db_util.py              # Database utility tests
│
├── 📁 utils/                        # Core Utility Modules
│   ├── __init__.py                  # Package initialization
│   ├── ai_util.py                   # OpenAI client management
│   ├── db_util.py                   # SQLite database operations
│   ├── vector_db_util.py            # Vector database operations
│   ├── xls_converter.py             # Excel data processing
│   ├── sql_chat.py                  # LangChain SQL agent
│   ├── simple_sql_chat.py           # Simplified SQL chat
│   ├── mock_sql_chat.py             # Mock SQL chat (testing)
│   ├── hybrid_rag_util.py           # Hybrid RAG implementation
│   └── preprocess_util.py           # Data preprocessing pipeline
│
├── 📄 Home.py                       # Main Streamlit application entry
├── 📄 main.py                       # FastAPI application (root level)
├── 📄 requirements.txt              # Python dependencies (version-free)
├── 📄 .env.sample                   # Environment variables template
├── 📄 .env                          # Environment configuration (local)
├── 📄 .gitignore                    # Git ignore patterns
├── 📄 PROJECT_SUMMARY.md            # Detailed project documentation
├── 📄 RENDER_DEPLOYMENT.md          # Render.com deployment guide
├── 📄 render.yaml                   # Render.com deployment configuration
└── 📄 README.md                     # This comprehensive guide
```

## 🚀 Quick Start Guide

### Prerequisites

- **Python 3.11+** (recommended)
- **OpenAI API Key** (for GPT-4o-mini)
- **Git** (for cloning the repository)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/kaljuvee/mli-rag-demo.git
cd mli-rag-demo

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.sample .env

# Edit .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 3. Launch Application

```bash
# Start Streamlit application
streamlit run Home.py

# Alternative: Start FastAPI backend
uvicorn main:app --reload --port 8000
```

### 4. Access the Application

- **Local Streamlit Web Interface**: http://localhost:8501
- **Local FastAPI Documentation**: http://localhost:8000/docs
- **Local API Health Check**: http://localhost:8000/health

- **Deployed Streamlit Web Interface**: https://mli-rag-demo.streamlit.app
- **Deployed API Documentation**: https://mli-rag-demo-api.onrender.com/docs
- **Deployed API Health Check**: https://mli-rag-demo-api.onrender.com/health

## 🎯 Core Features

### 1. 📊 Data Preprocessing Pipeline

**Location**: `pages/1_Preprocess.py`

- **Automated Excel Processing**: Loads and combines CurrentPortfolio.xlsx (1,250 properties) and MarketedWarehouses.xlsx (5 properties)
- **Data Cleaning & Validation**: Handles missing values, standardizes formats, validates data integrity
- **SQLite Database Creation**: Generates optimized database schema with proper indexing
- **Vector Database Initialization**: Creates property embeddings and FAISS index
- **Real-time Progress Feedback**: Visual confirmation of data loading with preview tables

**Key Capabilities**:
- Intelligent data type inference and conversion
- Missing value imputation strategies
- Duplicate detection and handling
- Coordinate validation and geocoding support

### 2. 🤖 AI-Powered Text-to-SQL

**Location**: `pages/2_Text-to-SQL.py`

Transform natural language questions into executable SQL queries using advanced language models.

**Example Queries**:
```
"Find the 10 most similar properties to the newly marketed property"
→ Complex similarity analysis with size and age metrics

"Provide a correlation score for homogeneity of marketed properties"
→ Multi-dimensional correlation analysis across physical, location, and age characteristics

"Find closest properties excluding those >10 miles from major cities"
→ Geographic proximity analysis with Haversine distance calculations
```

**Technical Implementation**:
- **LangChain SQL Agent**: Sophisticated query planning and execution
- **Schema-Aware Generation**: Understands database structure and relationships
- **Error Handling & Validation**: Robust query validation and error recovery
- **Performance Optimization**: Query optimization and result caching

### 3. 🔍 RAG-Based Property Analysis

**Location**: `pages/3_RAG_Analysis.py`, `utils/vector_db_util.py`

Advanced vector-based analysis using Retrieval-Augmented Generation for property insights.

#### 🧠 RAG Implementation Details

The RAG system is implemented using a **dual-database architecture**:

1. **SQL Database** (`db_util.py`):
   - Handles structured data storage and retrieval
   - Supports traditional SQL queries and joins
   - Optimized for filtering and aggregation operations

2. **Vector Database** (`vector_db_util.py`):
   - Manages property embeddings and similarity search
   - Implements FAISS vector index for efficient retrieval
   - Persists embeddings to disk for fast loading

#### 🔢 Vector Embedding Process

1. **Text Generation**:
   ```python
   def create_property_text(property_row):
       """Create text representation of property for embedding."""
       text_parts = []
       text_parts.append(f"Industrial Estate: {property_row['industrial_estate_name']}")
       text_parts.append(f"Unit: {property_row['unit_name']}")
       text_parts.append(f"Region: {property_row['region']}")
       text_parts.append(f"Size: {property_row['size_sqm']:.0f} square meters")
       text_parts.append(f"Built in: {property_row['build_year']:.0f}")
       # Add other property characteristics...
       return ". ".join(text_parts) + "."
   ```

2. **TF-IDF Vectorization**:
   ```python
   # Initialize vectorizer
   self.vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
   
   # Generate embeddings
   self.property_vectors = self.vectorizer.fit_transform(property_texts).toarray()
   ```

3. **FAISS Index Creation**:
   ```python
   # Create FAISS index
   self.faiss_index = faiss.IndexFlatIP(self.property_vectors.shape[1])
   
   # Normalize vectors for cosine similarity
   normalized_vectors = self.property_vectors / np.linalg.norm(
       self.property_vectors, axis=1, keepdims=True
   )
   
   # Add vectors to index
   self.faiss_index.add(normalized_vectors.astype('float32'))
   ```

4. **Persistence Layer**:
   ```python
   # Save vectorizer
   with open(self.vectorizer_path, 'wb') as f:
       pickle.dump(self.vectorizer, f)
   
   # Save property vectors
   np.save(self.vectors_path, self.property_vectors)
   
   # Save FAISS index
   faiss.write_index(self.faiss_index, self.faiss_index_path)
   ```

#### 🔍 Similarity Search Implementation

```python
def find_similar_properties(self, target_property_id=None, query_text=None, top_k=10):
    """Find similar properties using vector similarity."""
    
    # Get target vector (from property ID or text query)
    if target_property_id is not None:
        # Find property by ID and get its vector
        target_vector = self.property_vectors[property_idx]
    elif query_text:
        # Create embedding for query text
        target_vector = self.vectorize_query(query_text)
    
    # Normalize target vector
    target_vector = target_vector / np.linalg.norm(target_vector)
    
    # Search using FAISS
    similarities, indices = self.faiss_index.search(
        target_vector.reshape(1, -1).astype('float32'), 
        top_k
    )
    
    # Return results with property details
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
        "results": results,
        "avg_similarity": float(np.mean(similarities[0])),
        # Additional metadata...
    }
```

#### 📊 Portfolio Homogeneity Analysis

The system calculates portfolio homogeneity using vector similarity:

```python
def calculate_portfolio_homogeneity(self):
    """Calculate portfolio homogeneity using embeddings."""
    
    # Calculate pairwise similarities using normalized vectors
    normalized_vectors = self.property_vectors / np.linalg.norm(
        self.property_vectors, axis=1, keepdims=True
    )
    similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
    
    # Remove diagonal (self-similarities)
    np.fill_diagonal(similarity_matrix, 0)
    
    # Calculate overall metrics
    avg_similarity = np.mean(similarity_matrix)
    std_similarity = np.std(similarity_matrix)
    
    # Analyze marketed vs non-marketed properties
    # ...
    
    # Calculate homogeneity coefficient
    homogeneity_coefficient = avg_similarity / (std_similarity + 1e-8)
    
    return {
        "overall_homogeneity": float(avg_similarity),
        "homogeneity_coefficient": float(homogeneity_coefficient),
        "marketed_vs_portfolio": float(marketed_vs_portfolio),
        # Additional metrics...
    }
```

#### 🔄 Integration with Preprocessing

The vector database is initialized during preprocessing:

```python
def load_data_to_database(self, df: pd.DataFrame) -> bool:
    """Load cleaned data into database."""
    try:
        # Load data to SQL database
        # ...
        
        # Initialize vector database
        print("🔄 Initializing vector database...")
        vector_db.initialize_vectors(force_rebuild=True)
        
        return True
    except Exception as e:
        print(f"❌ Error loading data to database: {e}")
        return False
```

**Core Capabilities**:
- **Property Similarity Search**: Find similar properties using multi-dimensional embeddings
- **Portfolio Clustering**: Identify distinct property groups and market segments
- **Homogeneity Analysis**: Measure portfolio consistency and diversification
- **Contextual AI Responses**: Generate insights with property-specific context

**Vector Analysis Features**:
- **TF-IDF Embeddings**: Efficient text-based property representations
- **FAISS Vector Search**: Efficient similarity search and clustering
- **Cosine Similarity Metrics**: Precise similarity scoring
- **Persistent Vector Storage**: Fast loading with disk-based storage

### 4. 🗄️ Database Management

**Location**: `utils/db_util.py`, `utils/vector_db_util.py`

The application uses a dual-database architecture:

#### SQL Database (`PropertyDatabase` class)

```python
class PropertyDatabase:
    """Database utility for MLI property data."""
    
    def __init__(self, db_path=None):
        if db_path is None:
            # Default path: db/mli.db relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_dir = os.path.join(project_root, 'db')
            os.makedirs(db_dir, exist_ok=True)
            self.db_path = os.path.join(db_dir, 'mli.db')
        else:
            self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.conn = None
        self.engine = None
    
    # Database methods...
```

#### Vector Database (`PropertyVectorDB` class)

```python
class PropertyVectorDB:
    """Vector database for property data using FAISS."""
    
    def __init__(self, embeddings_dir=None):
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
        
        # Initialize components...
```

**Database Schema**:
```sql
CREATE TABLE properties (
    property_id REAL,
    industrial_estate_name TEXT,
    unit_name TEXT,
    region TEXT,
    latitude REAL,
    longitude REAL,
    car_parking_spaces INTEGER,
    size_sqm REAL,
    build_year REAL,
    yard_depth_m REAL,
    min_eaves_m REAL,
    max_eaves_m REAL,
    doors INTEGER,
    epc_rating TEXT,
    is_marketed INTEGER
);
```

**Performance Features**:
- Optimized indexing for common queries
- Connection pooling and management
- Transaction safety and rollback support
- Query result caching

## 🧪 Testing Framework

### Comprehensive Unit Testing

The project includes extensive unit tests with JSON result validation:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python tests/test_sql_chat.py      # SQL functionality
python tests/test_rag_util.py      # RAG functionality
python tests/test_db_util.py       # Database operations
python tests/test_xls_converter.py # Data processing
```

### Test Coverage

- **SQL Chat Tests**: 6 test cases covering query generation and execution
- **RAG Analysis Tests**: 8 test cases covering vector operations and similarity search
- **Database Tests**: 4 test cases covering CRUD operations and schema validation
- **Data Processing Tests**: 3 test cases covering Excel parsing and transformation

### JSON Test Results

All test results are saved as structured JSON files in `test-data/` for:
- **Result Validation**: Verify query outputs against expected results
- **Performance Monitoring**: Track query execution times and accuracy
- **Regression Testing**: Ensure consistent behavior across updates
- **Documentation**: Provide examples of expected system behavior

## 🔧 API Documentation

### FastAPI Endpoints

**Location**: `main.py`, `api/swagger.json`

```python
# Health Check
GET /health
→ {"status": "healthy", "timestamp": "2025-09-18T10:30:00Z"}

# Property Search
POST /properties/search
Body: {"query": "Find properties in London", "limit": 10}
→ {"results": [...], "count": 10, "execution_time": 0.15}

# Similarity Analysis
POST /properties/similar
Body: {"property_id": 123, "top_k": 5}
→ {"similar_properties": [...], "similarity_scores": [...]}

# RAG Query
POST /rag/query
Body: {"question": "What is the portfolio homogeneity?"}
→ {"answer": "...", "context": [...], "confidence": 0.95}
```

The API is fully documented using OpenAPI/Swagger:
- **Interactive Documentation**: Available at `/docs` endpoint
- **OpenAPI Specification**: Available at `/openapi.json` endpoint
- **Swagger JSON**: Included in the repository at `api/swagger.json`

## 📈 Performance Metrics

### Query Performance

- **Text-to-SQL Generation**: ~2-3 seconds per query
- **Vector Similarity Search**: ~0.1-0.5 seconds for 1,255 properties
- **Database Queries**: ~0.01-0.1 seconds for most operations
- **RAG Context Generation**: ~1-2 seconds including embeddings

### Scalability

- **Database**: Tested with 1,255 properties, scales to 100K+ with indexing
- **Vector Search**: FAISS enables sub-second search on millions of vectors
- **Concurrent Users**: Streamlit supports 10+ concurrent users
- **Memory Usage**: ~200MB baseline, ~500MB with full vector cache

## 🛠️ Development Guide

### Adding New Features

1. **New Query Types**: Extend `utils/sql_chat.py` with additional query patterns
2. **Vector Analysis**: Add new similarity metrics in `utils/vector_db_util.py`
3. **UI Components**: Create new pages in `pages/` directory
4. **API Endpoints**: Extend `main.py` with new routes

### Code Quality Standards

- **Type Hints**: All functions include comprehensive type annotations
- **Documentation**: Docstrings for all classes and methods
- **Error Handling**: Robust exception handling with user-friendly messages
- **Testing**: Unit tests required for all new functionality

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...                    # OpenAI API key for GPT and embeddings

# Optional
DATABASE_URL=sqlite:///db/mli.db         # Database connection string
LOG_LEVEL=INFO                           # Logging verbosity
CACHE_TTL=3600                           # Cache time-to-live in seconds
MAX_QUERY_TIME=30                        # Maximum query execution time
```

## 🚀 Deployment on Render.com

The MLI RAG Demo is designed for easy deployment on [Render.com](https://render.com), a simple and affordable cloud platform.

### Deployment Options

1. **Blueprint Deployment** (Recommended):
   - Use the included `render.yaml` file for automatic configuration
   - Render will create both the API and Streamlit services

2. **Manual Deployment**:
   - Deploy the API and Streamlit app separately through the Render Dashboard
   - Configure each service with the appropriate build and start commands

### Quick Deployment Steps

1. Fork or clone the repository to your GitHub account
2. Sign up for a [Render.com](https://render.com) account
3. In your Render Dashboard, click **New** and select **Blueprint**
4. Connect your GitHub repository
5. Add your `OPENAI_API_KEY` when prompted
6. Click **Apply** to deploy both services

For detailed deployment instructions, see [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md).

## 📊 Data Sources

### Current Portfolio (1,250 properties)
- **Source**: CurrentPortfolio.xlsx
- **Columns**: Property ID, Estate Name, Unit, Region, Coordinates, Size, Build Year, Physical Characteristics
- **Coverage**: UK-wide industrial properties across multiple regions

### Marketed Warehouses (5 properties)
- **Source**: MarketedWarehouses.xlsx
- **Properties**: Cherry Lane, Tech Hub, Spitfire Park, Stable Lane, Chancery Depot
- **Purpose**: Target properties for similarity analysis and portfolio comparison

## 🤝 Contributing

### Development Workflow

1. **Fork the repository** and create a feature branch
2. **Implement changes** following code quality standards
3. **Add unit tests** for new functionality
4. **Update documentation** including README and docstrings
5. **Submit pull request** with detailed description

### Issue Reporting

Please use GitHub Issues for:
- 🐛 Bug reports with reproduction steps
- 💡 Feature requests with use case descriptions
- 📚 Documentation improvements
- ❓ Questions about implementation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT-4o-mini models
- **LangChain** for SQL agent framework
- **Streamlit** for rapid web application development
- **FAISS** for efficient vector similarity search
- **MLI Assessment Team** for the comprehensive evaluation framework

## 📞 Support

For questions, issues, or contributions:

- **GitHub Issues**: [Create an issue](https://github.com/kaljuvee/mli-rag-demo/issues)
- **Documentation**: See `PROJECT_SUMMARY.md` for detailed technical documentation
- **Test Results**: Check `test-data/` directory for validation examples

---

**Built with ❤️ for the MLI Assessment - Demonstrating Production-Ready AI Engineering**
