#!/bin/bash
# Run evaluation to fill performance comparison table

echo "=================================================="
echo "  Triple-Stream Fashion Search Evaluation"
echo "=================================================="
echo ""
echo "This script will:"
echo "  1. Run all 5 test queries"
echo "  2. Compare Triple-Stream vs Vanilla CLIP"
echo "  3. Calculate Precision@10 for each query type"
echo "  4. Generate performance comparison table"
echo ""
echo "Method: Automatic relevance judgment using keyword matching"
echo "Time: ~2-3 minutes for 5 queries"
echo ""

# Ensure logs directory exists
mkdir -p logs

# Navigate to retriever directory and run evaluation
cd retriever
echo "Starting evaluation..."
python evaluate.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "  ✓ Evaluation Complete!"
    echo "=================================================="
    echo ""
    echo "Results saved to: evaluation_results.json"
    echo ""
    echo "Next steps:"
    echo "  1. Review evaluation_results.json for detailed results"
    echo "  2. Update README.md with the comparison table"
    echo "  3. (Optional) Run with visualizations: python evaluate.py --visualize"
    echo ""
else
    echo ""
    echo "❌ Evaluation failed. Check logs/evaluator.log for details"
    exit 1
fi
