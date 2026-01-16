#!/bin/bash
# retriever_pipeline.sh - Run retriever search queries

echo "=================================================="
echo "Glance Retriever Pipeline"
echo "=================================================="
echo ""

# Activate your environment (adjust as needed for DGX)
# conda activate glance_env  # Uncomment if using conda
# source venv/bin/activate   # Uncomment if using venv

cd /workspace  # Adjust to your project path

echo "Step 1: Testing Retriever Initialization..."
python3 -c "
from retriever.retriever import TripleStreamRetriever
import sys
sys.path.insert(0, '/workspace')

print('Loading retriever...')
retriever = TripleStreamRetriever()
print('✓ Retriever initialized successfully!')
print(f'✓ Loaded {len(retriever.collections)} collections')
"

if [ $? -ne 0 ]; then
    echo "❌ Retriever initialization failed!"
    exit 1
fi

echo ""
echo "Step 2: Running Search Queries..."
python3 retriever/retriever.py

echo ""
echo "=================================================="
echo "Retriever Pipeline Complete!"
echo "=================================================="