# My Simple Dashboard (Finance Streamlit Dashboard)

A minimal Streamlit dashboard template that generates example time-series data and provides basic filters, charts, and data download functionality. This project is intended as a starting point for visualizing financial or time-series datasets.

---

## Features

- Interactive date range filter and category selector in the sidebar.
- Metric selector to switch between `value` and `volume`.
- Top-level metrics (Total, Average, Last-day comparison).
- Tabs for Overview (time-series and category charts), Details (aggregations + CSV download), and Raw Data.
- Example data generator in `load_data()` — replace with your own data source.
- Caching via `@st.cache_data` for faster reloads.

---

## Prerequisites

- Python 3.8 or newer
- Windows PowerShell (instructions below assume PowerShell)

---

## Install

From the project root (`c:\Users\Owner\Documents\finance_streamlit_dashboard`):

1. Create and activate a virtual environment (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can run (as Administrator) to allow local scripts:

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

2. Upgrade `pip` and install dependencies:

```
python -m pip install --upgrade pip
pip install streamlit pandas numpy
```

Optionally, create a `requirements.txt`:

```
pip freeze > requirements.txt
```

---

## Run the app

From the project root (with the venv activated):

```
streamlit run app.py
```

Streamlit will print a local URL (e.g., `http://localhost:8501`) — open it in your browser.

---

## How it works / Where to customize

- `app.py` contains all the UI and example data generation.
- Replace the `load_data()` function with your own data loader (e.g., from CSV, database, or API).
- Charts use Streamlit's built-in `line_chart` and `bar_chart`. Replace with Altair or Plotly for more control.
- The date column is converted to pandas `datetime` for reliable filtering (`df['date'] = pd.to_datetime(df['date'])`).
- Caching is applied to `load_data()` via `@st.cache_data` to avoid reloading on every interaction.

Helpful places to edit:
- Sidebar controls: search for `st.sidebar` to add filters or inputs.
- Metrics and tabs: near the top-level metrics and `st.tabs(...)` blocks.

---

## Example usage notes

- Use the sidebar to select a date range and categories to filter the dashboard.
- The "Metric to visualize" selector toggles between `value` and `volume` and updates all charts/metrics.
- Use the Download button under the Details tab to export the filtered dataset as CSV.

---

## Troubleshooting

- `ModuleNotFoundError: No module named 'streamlit'` — ensure the venv is activated and run `pip install streamlit`.
- Streamlit port already in use — stop the other instance or run `streamlit run app.py --server.port 8502`.
- PowerShell script execution prevented — enable local scripts with `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

---

## Contributing

Feel free to open issues or submit pull requests. Suggestions:

- Add a `requirements.txt` or `pyproject.toml` for reproducible installs.
- Replace the fake data generator with connectors to real data sources.
- Add unit tests or a lightweight data validation step for inputs.

---

## License

Add a license file (e.g., `LICENSE`) if you plan to publish this project. No license is included by default.

---

## Contact

If you'd like changes to this README (more examples, badges, or CI instructions), tell me what you'd like added and I will update it.
