# MLI RAG Demo - System Architecture Diagrams

This directory contains architectural diagrams for the MLI RAG Demo application. The diagrams are created using Mermaid, a markdown-based diagramming tool.

## System Architecture

The overall system architecture of the MLI RAG Demo:

```mermaid
flowchart TB
    subgraph "Frontend"
        UI[Streamlit UI]
        UI --> |User Queries| Home[Home.py]
        Home --> Preprocess[1_Preprocess.py]
        Home --> TextToSQL[2_Text-to-SQL.py]
        Home --> RAGAnalysis[3_RAG_Analysis.py]
        Home --> SimpleQuery[4_Simple_Query.py]
    end
    
    subgraph "Backend API"
        API[FastAPI]
        API --> |Health Check| Health[/health]
        API --> |Property Data| PropAPI[/properties]
        API --> |SQL Queries| SQLAPI[/properties/search]
        API --> |RAG Analysis| RAGAPI[/rag]
    end
    
    subgraph "Core Utilities"
        Utils[Utils]
        Utils --> DBUtil[db_util.py]
        Utils --> VectorDBUtil[vector_db_util.py]
        Utils --> AIUtil[ai_util.py]
        Utils --> SQLChat[sql_chat.py]
        Utils --> PreprocessUtil[preprocess_util.py]
    end
    
    subgraph "Data Storage"
        Storage[Storage]
        Storage --> SQLite[(SQLite DB)]
        Storage --> FAISS[(FAISS Vector DB)]
        Storage --> Excel[(Excel Files)]
    end
    
    subgraph "External Services"
        External[External]
        External --> OpenAI[OpenAI API]
    end
    
    UI --> API
    TextToSQL --> SQLChat
    RAGAnalysis --> VectorDBUtil
    Preprocess --> PreprocessUtil
    PreprocessUtil --> DBUtil
    PreprocessUtil --> VectorDBUtil
    DBUtil --> SQLite
    VectorDBUtil --> FAISS
    SQLChat --> AIUtil
    AIUtil --> OpenAI
    PreprocessUtil --> Excel
```

## RAG Flow

The sequence diagram for the RAG Analysis process:

```mermaid
sequenceDiagram
    participant User
    participant RAGPage as RAG Analysis Page
    participant VectorDB as Vector Database
    participant FAISS as FAISS Index
    participant SQLite as SQLite Database
    
    User->>RAGPage: Select Query Type
    User->>RAGPage: Select Target Property
    User->>RAGPage: Click "Run Analysis"
    
    alt Find Similar Properties
        RAGPage->>VectorDB: get_property_vector(property_id)
        VectorDB->>FAISS: search(property_vector, top_k=10)
        FAISS-->>VectorDB: similar_indices, similarity_scores
        VectorDB->>SQLite: get_properties_by_ids(property_ids)
        SQLite-->>VectorDB: property_details
        VectorDB-->>RAGPage: similar_properties_with_scores
        RAGPage-->>User: Display Similar Properties Table
    
    else Calculate Portfolio Homogeneity
        RAGPage->>VectorDB: calculate_portfolio_homogeneity()
        VectorDB->>FAISS: get_all_vectors()
        FAISS-->>VectorDB: all_property_vectors
        VectorDB->>VectorDB: compute_similarity_matrix()
        VectorDB->>VectorDB: calculate_metrics()
        VectorDB-->>RAGPage: homogeneity_metrics
        RAGPage-->>User: Display Homogeneity Analysis
    
    else Find Properties Near Cities
        RAGPage->>VectorDB: get_property_vector(property_id)
        VectorDB->>SQLite: get_properties_near_cities(max_distance=10)
        SQLite-->>VectorDB: filtered_properties
        VectorDB->>FAISS: search_filtered(property_vector, filtered_ids)
        FAISS-->>VectorDB: similar_indices, similarity_scores
        VectorDB-->>RAGPage: nearby_similar_properties
        RAGPage-->>User: Display Nearby Properties Table
    end
```

## Text-to-SQL Flow

The sequence diagram for the Text-to-SQL process:

```mermaid
sequenceDiagram
    participant User
    participant SQLPage as Text-to-SQL Page
    participant SQLChat as SQL Chat Utility
    participant OpenAI as OpenAI API
    participant DBUtil as Database Utility
    participant SQLite as SQLite Database
    
    User->>SQLPage: Enter Natural Language Query
    User->>SQLPage: Click "Get Answer"
    
    SQLPage->>DBUtil: get_schema()
    DBUtil->>SQLite: PRAGMA table_info(properties)
    SQLite-->>DBUtil: schema_info
    DBUtil-->>SQLPage: formatted_schema
    
    SQLPage->>SQLChat: get_sql_from_prompt(query, schema)
    SQLChat->>OpenAI: generate_completion(system_prompt + query + schema)
    OpenAI-->>SQLChat: generated_sql
    SQLChat-->>SQLPage: sql_query
    
    SQLPage->>DBUtil: execute_query(sql_query)
    DBUtil->>SQLite: execute(sql_query)
    SQLite-->>DBUtil: query_results
    DBUtil-->>SQLPage: results_dataframe
    
    SQLPage-->>User: Display SQL Query
    SQLPage-->>User: Display Results Table
    SQLPage-->>User: Display Raw JSON
```

## Preprocessing Flow

The flowchart for the data preprocessing pipeline:

```mermaid
flowchart TD
    Start([Start Preprocessing]) --> LoadExcel[Load Excel Files]
    LoadExcel --> |CurrentPortfolio.xlsx| CurrentDF[Current Properties DataFrame]
    LoadExcel --> |MarketedWarehouses.xlsx| MarketedDF[Marketed Properties DataFrame]
    
    CurrentDF --> CleanCurrent[Clean Current Properties]
    MarketedDF --> CleanMarketed[Clean Marketed Properties]
    
    CleanCurrent --> |Handle Missing Values| CurrentClean[Cleaned Current Properties]
    CleanMarketed --> |Handle Missing Values| MarketedClean[Cleaned Marketed Properties]
    
    CurrentClean --> Merge[Merge DataFrames]
    MarketedClean --> Merge
    
    Merge --> |Combined DataFrame| CreateDB[Create SQLite Database]
    CreateDB --> |Create Tables| LoadDB[Load Data to Database]
    
    LoadDB --> |Database Populated| InitVectors[Initialize Vector Database]
    InitVectors --> |Generate Property Texts| Vectorize[Create TF-IDF Vectors]
    Vectorize --> |Create Embeddings| CreateFAISS[Build FAISS Index]
    CreateFAISS --> |Save Index| SaveVectors[Save Vector Files]
    
    SaveVectors --> |Vectors Saved| Complete([Preprocessing Complete])
```

## API Flow

The sequence diagram for the API endpoints:

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI as FastAPI Server
    participant DBUtil as Database Utility
    participant VectorDB as Vector Database
    participant SQLite as SQLite Database
    
    Client->>FastAPI: GET /health
    FastAPI-->>Client: {"status": "healthy", "timestamp": "..."}
    
    Client->>FastAPI: GET /properties?limit=10&offset=0
    FastAPI->>DBUtil: query_to_dict("SELECT * FROM properties LIMIT 10")
    DBUtil->>SQLite: execute query
    SQLite-->>DBUtil: results
    DBUtil-->>FastAPI: properties_list
    FastAPI-->>Client: {"success": true, "properties": [...]}
    
    Client->>FastAPI: GET /properties/123
    FastAPI->>DBUtil: query_to_dict("SELECT * FROM properties WHERE property_id = ?", [123])
    DBUtil->>SQLite: execute query
    SQLite-->>DBUtil: property_data
    DBUtil-->>FastAPI: property_dict
    FastAPI-->>Client: {"success": true, "property": {...}}
    
    Client->>FastAPI: POST /properties/search {"query": "SELECT * FROM properties WHERE region = 'North West'"}
    FastAPI->>DBUtil: query_to_dict(request.query)
    DBUtil->>SQLite: execute query
    SQLite-->>DBUtil: results
    DBUtil-->>FastAPI: properties_list
    FastAPI-->>Client: {"success": true, "properties": [...]}
    
    Client->>FastAPI: GET /properties/marketed
    FastAPI->>DBUtil: query_to_dict("SELECT * FROM properties WHERE is_marketed = 1")
    DBUtil->>SQLite: execute query
    SQLite-->>DBUtil: marketed_properties
    DBUtil-->>FastAPI: properties_list
    FastAPI-->>Client: {"success": true, "properties": [...]}
```

## Vector Database Class Diagram

The class diagram for the vector database implementation:

```mermaid
classDiagram
    class PropertyVectorDB {
        -embeddings_dir: str
        -vectors_path: str
        -faiss_index_path: str
        -vectorizer_path: str
        -property_ids_path: str
        -property_vectors: ndarray
        -faiss_index: IndexFlatIP
        -vectorizer: TfidfVectorizer
        -property_ids: List[int]
        +__init__(embeddings_dir=None)
        +load_vectors()
        +initialize_vectors(force_rebuild=False)
        +check_initialized()
        +create_property_text(property_row)
        +vectorize_query(query_text)
        +find_similar_properties(target_property_id, query_text, top_k)
        +calculate_portfolio_homogeneity()
        +get_property_by_id(property_id)
        +get_properties_near_cities(max_distance_miles)
    }
    
    class FAISS {
        +IndexFlatIP
        +normalize_L2(vectors)
        +add(vectors)
        +search(query_vector, k)
        +write_index(index, path)
        +read_index(path)
    }
    
    class TfidfVectorizer {
        +max_features: int
        +stop_words: str
        +fit_transform(texts)
        +transform(query)
    }
    
    class PropertyDatabase {
        -db_path: str
        -conn: Connection
        -engine: Engine
        +__init__(db_path=None)
        +connect()
        +close()
        +query_to_df(query, params=None)
        +execute_query(query, params=None)
        +get_engine()
        +get_schema()
    }
    
    PropertyVectorDB --> FAISS : uses
    PropertyVectorDB --> TfidfVectorizer : uses
    PropertyVectorDB --> PropertyDatabase : queries
```
