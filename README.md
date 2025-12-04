# Finance Streamlit Dashboard

This repository contains a Streamlit dashboard template for exploring time-series and event-driven financial data. There are two Streamlit apps in the repo:

- `app.py` — a simple template dashboard that generates example time-series data and demonstrates filters, charts, and CSV export.
- `storage_old.py` (and other supporting files) — a more advanced "Senator Trades Event Study" dashboard that loads `results_df.csv`, computes event-study metrics, fetches news via Marketaux, and can summarize news using the Google Gemini API.

Use this README to get the app running locally, understand the required data/keys, and customize the dashboards.

**Key files**
- `app.py`: Simple example dashboard (time-series generator, sidebar controls, charts, CSV download).
- `storage_old.py`: Senator Trades Event Study dashboard (market event processing, Marketaux news fetcher, Gemini summarizer).
- `results_df.csv`: Required by the event-study app (`storage_old.py`) — contains event/windowed returns and AR/CAR fields.
- `requirements.txt`: List of Python dependencies used by the project.

**Features (combined)**
- Interactive sidebar filters (date range, categories, party, senator, event window).
- Time-series and bar charts, aggregated views, raw data exploration, and CSV download.
- Event-study computation (AR/CAR/SCAR) and per-event cumulative return metrics.
- News fetching via Marketaux and optional summarization via Google Gemini (requires API keys).

**Prerequisites**
- Python 3.8+
- Git (optional)

**Install (recommended)**
1. From the project root (`c:\Users\Owner\Documents\finance_streamlit_dashboard`), create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run (as Administrator) to allow local scripts:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer to install only the minimal packages for `app.py`, run:

```powershell
pip install streamlit pandas numpy
```

**Environment variables / API keys**
- `GEMINI_API_KEY`: (optional) Google Gemini API key used by `storage_old.py` for summarization. The app uses the `google-generativeai` package.
- `MARKETAUX_API_KEY`: (optional) Marketaux API key used to fetch news articles for tickers.

Set environment variables on Windows PowerShell like:

```powershell
$env:GEMINI_API_KEY = 'your_gemini_key_here'
$env:MARKETAUX_API_KEY = 'your_marketaux_key_here'
```

To persist them across sessions, add them to your system environment variables or your PowerShell profile.

**Run the dashboards**

To run the simple example app:

```powershell
streamlit run app.py
```

To run the event-study / news summarizer app (if you have `results_df.csv` and API keys):

```powershell
streamlit run storage_old.py
```

Streamlit will print a local URL (e.g., `http://localhost:8501`) — open it in your browser.

**Data: `results_df.csv`**
- Place `results_df.csv` in the project root (next to `app.py`) to use the Senator Trades event-study app. The app expects columns like `event_id`, `ticker`, `event_date`, `Offset`, `Ret`, and optionally `ar`, `car`, `scar`.

**Notes on customization**
- Replace the example data generator in `app.py` (`load_data`) with your own data loader (CSV, database, API).
- The event-study logic in `storage_old.py` computes per-event cumulative returns and aggregates by senator; edit the grouping window or metrics as required.
- Charts are Streamlit-native; swap to Altair/Plotly if you need more advanced visuals.

**Troubleshooting**
- `ModuleNotFoundError: No module named 'streamlit'` — ensure the venv is activated and run `pip install -r requirements.txt`.
- Streamlit port already in use — stop the other instance or run `streamlit run app.py --server.port 8502`.
- If Marketaux or Gemini calls fail, ensure the corresponding API keys are set and have valid credits/permissions.

**Contributing & Next steps**
- Add a `LICENSE` file if you plan to publish the project.
- Add tests or a sample `results_df.csv` with synthetic data for easier local testing.
- Consider splitting the large Streamlit app into smaller modules and adding a `procfile` / Dockerfile for deployment.

If you want, I can:
- add a small example `results_df.csv` with synthetic data,
- create a `Dockerfile` or GitHub Actions workflow to deploy the app,
- or update `requirements.txt` to pin specific versions.

Tell me which of these you'd like next.
