#!/bin/bash

# Weight Optimization Script
# Runs Bayesian optimization to find optimal alpha, beta, gamma weights

echo "========================================"
echo "Weight Optimization - Bayesian Search"
echo "========================================"
echo ""

# Check if optuna is installed
if ! python -c "import optuna" 2>/dev/null; then
    echo "❌ Optuna not found. Installing..."
    pip install optuna==4.1.0
fi

# Get optimization mode
MODE=${1:-full}
GLOBAL_TRIALS=${2:-50}
PRESET_TRIALS=${3:-30}

echo "Configuration:"
echo "  Mode: $MODE"
echo "  Global trials: $GLOBAL_TRIALS"
echo "  Preset trials: $PRESET_TRIALS"
echo ""

# Run optimization
python retriever/optimize_weights.py \
    --mode "$MODE" \
    --global-trials "$GLOBAL_TRIALS" \
    --preset-trials "$PRESET_TRIALS"

echo ""
echo "========================================"
echo "Optimization Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - shared/config.yaml (updated weights)"
echo "  - outputs/weight_optimization_report.json"
echo ""
echo "Backup of original config saved with timestamp."
