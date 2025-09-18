# Test Data Directory

This directory contains JSON results from unit tests for the MLI Demo SQL chat functionality.

## JSON Test Result Files

### Core Query Tests
- **`query_1_similar_properties.json`** - Results for "Find the 10 most similar properties in the estate to the newly marketed property"
- **`query_2_correlation_score.json`** - Results for "Provide a correlation score for the homogeneity of the marketed property(ies) with the rest of the estate"
- **`query_3_closest_properties.json`** - Results for "Find the closest properties to the marketed property, after excluding any property in the portfolio that is more than 10 miles from a major city"

### Supporting Tests
- **`simple_queries.json`** - Results from basic functionality tests (marketed properties, count, average size)
- **`sql_generation_results.json`** - SQL generation validation for all 3 core queries

## JSON Structure

Each query result file contains:
```json
{
  "query": "Natural language question",
  "success": true/false,
  "sql_query": "Generated SQL query",
  "error": null or error message,
  "data_shape": [rows, columns],
  "sample_data": [first 3 rows of results],
  "column_names": ["list", "of", "column", "names"]
}
```

## Test Execution

To regenerate these JSON results:
```bash
cd /home/ubuntu/mli-rag-demo
PYTHONPATH=/home/ubuntu/mli-rag-demo python3 tests/test_sql_chat.py
```

## Validation

All tests pass (6/6) with:
- ✅ Database connectivity
- ✅ SQL query generation
- ✅ Data retrieval and formatting
- ✅ Error handling

**Status**: Production ready - all core functionality validated through unit tests.
