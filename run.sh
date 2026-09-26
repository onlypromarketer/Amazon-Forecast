#!/bin/zsh
# Start the Amazon Forecast dashboard at http://localhost:8502
cd "$(dirname "$0")"
exec .venv/bin/streamlit run app.py --server.port 8502 --browser.gatherUsageStats false
