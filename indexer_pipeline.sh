#!/bin/bash

# Run the complete indexing pipeline for the full dataset
# This script handles both caption generation and vector indexing

echo "=================================================="
echo "  Triple-Stream Fashion Search - Full Indexing"
echo "=================================================="
echo ""

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run: python -m venv venv"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "Step 1/2: Generating Vibe Captions (BLIP-2)"
echo "This will take ~6.5 hours for 45,623 images..."
echo "Progress will be saved every 500 images (checkpointed)"
echo ""

cd indexer
nohup python caption_generator.py > ../logs/caption_run.log 2>&1 &
CAPTION_PID=$!

echo "✓ Caption generation started (PID: $CAPTION_PID)"
echo "  Monitor: tail -f logs/caption_run.log"
echo ""
echo "Waiting for caption generation to complete..."

# Wait for caption generation to complete
wait $CAPTION_PID

if [ $? -eq 0 ]; then
    echo "✓ Caption generation completed successfully!"
else
    echo "❌ Caption generation failed. Check logs/caption_run.log"
    exit 1
fi

echo ""
echo "Step 2/2: Building Vector Index (ChromaDB)"
echo "This will take ~7-8 hours for grounded layer + indexing..."
echo ""

nohup python indexer.py > ../logs/indexer_run.log 2>&1 &
INDEXER_PID=$!

echo "✓ Indexing started (PID: $INDEXER_PID)"
echo "  Monitor: tail -f logs/indexer_run.log"
echo ""
echo "Waiting for indexing to complete..."

# Wait for indexing to complete
wait $INDEXER_PID

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "  ✓ Full Indexing Pipeline Complete!"
    echo "=================================================="
    echo ""
    echo "Collections created in: chroma_db/"
    echo "  - grounded_vectors: V_fact (Fashionpedia + colors)"
    echo "  - vibe_vectors: V_vibe (BLIP-2 captions)"
    echo "  - visual_vectors: V_img (CLIP image embeddings)"
    echo ""
    echo "You can now run searches with:"
    echo "  cd retriever && python retriever.py"
else
    echo "❌ Indexing failed. Check logs/indexer_run.log"
    exit 1
fi
