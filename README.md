# Data Cleaning and Viz
A lightweight, high-performance data cleaning and visualization tool built using Polars, Python, and Streamlit.
This project cleans messy stock-market CSV files, standardizes schema, computes aggregations (daily averages, volume summaries, returns), and provides an interactive UI for exploring the cleaned dataset.

## Features
- Automatic schema detection + type coercion (Polars)
- Cleans numeric columns (prices, volume, percentages)
- Cleans boolean and date columns
- Removes invalid tickers/sector values
- Computes:
 - Agg1: Daily average close price per ticker
 - Agg2: Average volume per sector
 - Agg3: Daily percent return per ticker
- Streamlit dashboard for:
 - Filtering tickers
 - Plotting price trends & returns
 - Viewing cleaned data and aggregations

## Installation of Required Libraries
Install dependencies:
```
uv sync
```

## Running the Application
Run the data cleaning CLI:
```
uv run main.py
```
This will load the dataset from data/, clean it, generate aggregations, and write output files.
Run the Streamlit Dashboard:
```
streamlit run app.py
```

## Library Requirements
```
polars
pyarrow
streamlit
```

## Project Structure
```
data_cleaning/
├─ data/                 # Input CSV files and Aggregated and Cleaned parquet files
├─ screenshots/          # Screenshots of demo
├─ main.py               # Data cleaning pipeline
├─ app.py                # Interactive UI
├─ README.md             # Project documentation
├─ .python-version
├─ uv.lock
├─ .gitignore
└─ pyproject.toml
```

## Additional Usage
Run Streamlit in development mode (auto-reload):
```
streamlit run app.py --server.runOnSave true
```

## Contribution
- Pull requests and suggestions are welcome.
- Please open an issue before submitting large changes.

