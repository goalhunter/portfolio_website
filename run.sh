#!/bin/bash
source venv/bin/activate
pip uninstall pinecone-client
pip install -r requirements.txt
python -m streamlit run streamlit_app.py --browser.gatherUsageStats False --server.port 8000 --server.address 0.0.0.0