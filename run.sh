#!/bin/bash
source venv/bin/activate
python -m streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
