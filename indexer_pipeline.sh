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

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/logs"

echo "Running Integrated Pipeline:"
echo "  Phase 1: Caption Generation (caption_generator.py)"
echo "    - Generate grounded vectors (V_fact)"
echo "    - Generate context-aware captions (V_vibe)"
echo "  Phase 2: Vectorization & Indexing (indexer.py)"
echo "    - Index grounded layer"
echo "    - Index vibe layer"
echo "    - Index visual layer"
echo ""
echo "Estimated time: ~5-6 hours for 45,623 images (Florence-2 5x faster!)"
echo "Progress saved every 500 images (checkpointed)"
echo ""

# Phase 1: Caption Generation
echo "=================================================="
echo "  PHASE 1: Caption Generation"
echo "=================================================="
cd indexer
nohup python caption_generator.py > ../logs/caption_run.log 2>&1 &
CAPTION_PID=$!

echo "✓ Caption generation started (PID: $CAPTION_PID)"
echo "  Stages:"
echo "    [1/2] Grounded vector generation (~2-3 hours)"
echo "    [2/2] Florence-2 captioning (~1 hour, 5x faster!)"
echo "  Monitor: tail -f logs/caption_run.log"
echo ""
echo "Waiting for caption generation to complete..."

wait $CAPTION_PID

if [ $? -ne 0 ]; then
    echo "❌ Caption generation failed. Check logs/caption_run.log"
    exit 1
fi

echo "✓ Caption generation complete!"
echo ""

# Phase 2: Indexing
echo "=================================================="
echo "  PHASE 2: Vector Indexing"
echo "=================================================="
nohup python indexer.py > ../logs/indexer_run.log 2>&1 &
INDEXER_PID=$!

echo "✓ Indexing started (PID: $INDEXER_PID)"
echo "  Stages:"
echo "    [1/3] Grounded layer indexing (~15 min)"
echo "    [2/3] Vibe layer indexing (~15 min)"
echo "    [3/3] Visual layer indexing (~2-3 hours)"
echo "  Monitor: tail -f logs/indexer_run.log"
echo ""
echo "Waiting for indexing to complete..."

# Wait for indexing to complete
wait $INDEXER_PID

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "  ✓ Integrated Pipeline Complete!"
    echo "=================================================="
    echo ""
    echo "Phase 1 Output:"
    echo "  - grounded_vectors.json (V_fact metadata)"
    echo "  - vibe_captions.json (context-aware scene descriptions)"
    echo ""
    echo "Phase 2 Output:"
    echo "  ChromaDB collections in: chroma_db/"
    echo "    - grounded_vectors: V_fact (Fashionpedia + colors)"
    echo "    - vibe_vectors: V_vibe (context-aware Florence-2 captions)"
    echo "    - visual_vectors: V_img (CLIP image embeddings)"
    echo ""
    echo "You can now run searches with:"
    echo "  cd retriever && python retriever.py"
else
    echo "❌ Indexing failed. Check logs/indexer_run.log"
    exit 1
fi
