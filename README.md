# MLI Demo

This project is a proof-of-concept Streamlit application that demonstrates text-to-SQL and RAG-based analysis of Multi-Let Industrial (MLI) data.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/kaljuvee/mli-rag-demo.git
    cd mli-rag-demo
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**

    Create a `.env` file in the root directory and add your OpenAI API key:

    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run Home.py
    ```

2.  **Open your browser** and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Preprocess the data:**

    -   Navigate to the **Preprocess** page from the sidebar.
    -   Click the "Run Preprocessing" button to load, clean, and store the data in a local SQLite database.

4.  **Use Text-to-SQL:**

    -   Navigate to the **Text-to-SQL** page.
    -   Enter a question in natural language about the properties and click "Get Answer".

5.  **Use RAG Analysis:**

    -   Navigate to the **RAG Analysis** page.
    -   Select a marketed property from the dropdown and click "Find Similar Properties" to see a list of the most similar properties from the portfolio.

## API

The application also includes a FastAPI wrapper for the core functionalities. To run the API:

```bash
uvicorn api.main:app --reload
```

You can then send POST requests to the `/chat` endpoint with a JSON payload like:

```json
{
    "query": "Find the 10 largest properties by size",
    "type": "sql"
}
```

or

```json
{
    "query": "Find properties similar to Chancery Depot",
    "type": "rag"
}
```

