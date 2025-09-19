# Deploying MLI RAG Demo on Render.com

This guide provides step-by-step instructions for deploying the MLI RAG Demo application on Render.com, a simple and affordable cloud platform that's perfect for small to medium-sized applications.

## Prerequisites

1. A [Render.com](https://render.com) account (free tier available)
2. Your MLI RAG Demo code pushed to a GitHub repository
3. An OpenAI API key

## Deployment Options

You can deploy the MLI RAG Demo on Render.com in two ways:

1. **Manual Deployment**: Deploy the API and Streamlit app separately through the Render Dashboard
2. **Blueprint Deployment**: Use the `render.yaml` file for automatic configuration

## Option 1: Manual Deployment

### Deploying the FastAPI Backend

1. Log in to your [Render Dashboard](https://dashboard.render.com/)
2. Click **New** and select **Web Service**
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `mli-rag-demo-api` (or your preferred name)
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free or Standard (as needed)
5. Add environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PYTHON_VERSION`: `3.11.0`
6. Click **Create Web Service**

### Deploying the Streamlit Frontend

1. In your Render Dashboard, click **New** and select **Web Service**
2. Connect the same GitHub repository
3. Configure the service:
   - **Name**: `mli-rag-demo` (or your preferred name)
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run Home.py --server.port $PORT --server.address 0.0.0.0`
   - **Plan**: Free or Standard (as needed)
4. Add the same environment variables as the API service
5. Click **Create Web Service**

## Option 2: Blueprint Deployment (Recommended)

The repository includes a `render.yaml` file that defines both services for automatic deployment:

1. Log in to your [Render Dashboard](https://dashboard.render.com/)
2. Click **New** and select **Blueprint**
3. Connect your GitHub repository
4. Render will automatically detect the `render.yaml` file and configure both services
5. Add your `OPENAI_API_KEY` when prompted
6. Click **Apply** to deploy both services

## Post-Deployment Configuration

### Database Initialization

After deployment, you'll need to initialize the database:

1. Access your deployed Streamlit app (e.g., `https://mli-rag-demo.onrender.com`)
2. Navigate to the **Preprocess** page
3. Click **Run Preprocessing** to load the data and create the database

### API Documentation

The API documentation is available at:
- Swagger UI: `https://mli-rag-demo-api.onrender.com/docs`
- OpenAPI JSON: `https://mli-rag-demo-api.onrender.com/openapi.json`

## Persistent Storage

Render's free and standard plans provide ephemeral storage, which means files may be lost when services restart. For production use:

1. **Database Persistence**: Use the `/tmp` directory for the SQLite database as configured in the application
2. **Vector Database**: The application is configured to rebuild the vector database when needed

## Monitoring and Logs

1. Access your service in the Render Dashboard
2. Click on the **Logs** tab to view application logs
3. Set up **Custom Domains** in the settings if you want to use your own domain

## Scaling and Performance

1. Upgrade to a higher plan if you need more resources
2. Render automatically scales based on your plan
3. For better performance, consider upgrading from the free tier to a standard plan

## Troubleshooting

If you encounter issues:

1. Check the application logs in the Render Dashboard
2. Verify that all environment variables are set correctly
3. Ensure the preprocessing step has been completed
4. Check that the OpenAI API key is valid and has sufficient credits

## Cost Management

- The free tier is suitable for development and testing
- Standard plans start at $7/month per service
- Monitor your usage in the Render Dashboard to avoid unexpected charges

---

For more information, visit the [Render Documentation](https://render.com/docs) or contact the MLI Demo team for support.
