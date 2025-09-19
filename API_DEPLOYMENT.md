# MLI RAG Demo API Deployment Guide

This guide explains how to deploy the MLI RAG Demo API using a simple approach with minimal dependencies.

## Prerequisites

- Python 3.8+
- pip
- Git

## Deployment Steps

### 1. Clone the Repository

```bash
git clone https://github.com/kaljuvee/mli-rag-demo.git
cd mli-rag-demo
```

### 2. Install Dependencies

The API requires minimal dependencies:

```bash
pip install fastapi uvicorn pydantic
```

### 3. Run the API

Start the FastAPI application with:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

For production deployment, you might want to use:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Verify Deployment

The API should now be running. You can verify by accessing:

- Health check: `http://localhost:8000/health`
- API documentation: `http://localhost:8000/docs`

## Deployment on Render.com

To deploy on Render.com:

1. Create a new Web Service
2. Connect your GitHub repository
3. Configure the service:
   - **Build Command**: `pip install fastapi uvicorn pydantic`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## API Endpoints

The API provides the following endpoints:

- `GET /health`: Health check endpoint
- `GET /properties`: Get all properties with pagination
- `GET /properties/{property_id}`: Get a specific property by ID
- `GET /properties/marketed`: Get all marketed properties
- `POST /properties/search`: Search properties using SQL queries

## Database

The API uses a SQLite database located in the `db/` directory. The database is created during the preprocessing step in the Streamlit application.

## Troubleshooting

If you encounter issues:

1. **Database not found**: Make sure you've run the preprocessing step in the Streamlit app first
2. **Port already in use**: Change the port number in the uvicorn command
3. **Import errors**: Ensure you've installed the minimal dependencies listed above

## Notes

- This API implementation uses direct SQLite connections without ORM dependencies
- The API is designed to be lightweight and easy to deploy
- For more advanced features, additional dependencies may be required
