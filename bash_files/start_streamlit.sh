#!/bin/bash
# Streamlit launcher for DGX Docker environment
# Usage: ./start_streamlit.sh

echo "=================================================="
echo "  Glance Fashion Search - Streamlit Interface"
echo "=================================================="
echo ""
echo "Starting Streamlit on port 8501..."
echo ""
echo "Access the app from your browser:"
echo "  http://$(hostname -I | awk '{print $1}'):8501"
echo ""
echo "Press Ctrl+C to stop"
echo "=================================================="
echo ""

streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
