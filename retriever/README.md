# Retriever Module

## Overview
The retriever module provides an intelligent search engine that dynamically weights three vector streams based on query intent.

**Current Status**: ✅ **OPERATIONAL** - All collections loaded, latency tracking enabled

**Key Capabilities:**
- Dynamic query-time weighting: `α·S_fact + β·S_vibe + γ·S_img`
- 5 pre-configured weight presets for different query types
- Query expansion with fashion-specific synonyms
- Color-based re-ranking
- Attribute filtering (min/max garments)
- Batch metadata retrieval (10-30x faster)
- Real-time latency tracking (~30-40ms per query)

**Performance**: Searching 45,623 images across 3 collections in ~35ms average

## Components

### 1. `retriever.py` - Triple-Stream Search Engine

**Status**: ✅ Fully operational with optimizations

**Key Features:**
- **Dynamic Weighting**: Query-time fusion with configurable α, β, γ weights
- **Query Expansion**: Automatic synonym expansion ("yellow" → "yellow vibrant sunny golden")
- **Weight Presets**: 5 pre-configured presets optimized for different query types
- **Batch Metadata**: Parallel metadata fetching (10-30x faster than sequential)
- **Color Re-ranking**: Post-retrieval boosting for color matches
- **Attribute Filtering**: Filter by garment count (min/max)
- **Latency Tracking**: Real-time performance monitoring
- **Score Normalization**: Three methods (relative, exponential, inverse)

**Measured Performance (DGX Server):**
- First query: ~234ms (model warmup)
- Subsequent queries: ~30-40ms average
- Collections: 3 × 45,623 vectors
- Default method: `relative` (most stable)

**Query Types & Presets:**

| Preset | Example Query | α (Grounded) | β (Vibe) | γ (Visual) |
|--------|---------------|--------------|----------|------------|
| `attribute_specific` | "bright yellow raincoat" | 0.4 | 0.1 | 0.5 |
| `contextual_place` | "modern office attire" | 0.3 | 0.6 | 0.1 |
| `complex_semantic` | "blue shirt on park bench" | 0.3 | 0.5 | 0.2 |
| `style_inference` | "casual weekend city walk" | 0.2 | 0.7 | 0.1 |
| `compositional` | "red tie + white shirt" | 0.4 | 0.1 | 0.5 |

**Usage:**
```python
from retriever.retriever import TripleStreamRetriever

# Initialize (loads CLIP model + 3 ChromaDB collections)
retriever = TripleStreamRetriever()

# Basic search with preset
results = retriever.dynamic_search(
    query="A person in a bright yellow raincoat",
    preset="attribute_specific",
    top_k=10
)
# Logs: ⏱️ Retrieval Latency: 35.42 ms

# Advanced search with custom weights and filters
results = retriever.dynamic_search(
    query="Casual weekend outfit",
    alpha=0.2,      # Grounded attributes
    beta=0.7,       # Vibe/context  
    gamma=0.1,      # Visual similarity
    top_k=10,
    expand=True,    # Query expansion
    filters={'min_garments': 3},  # At least 3 garments
    score_method='relative'  # Normalization method
)

# Apply color re-ranking (boosts color matches)
results = retriever.rerank_by_color(results, query, boost_factor=1.3)

# Print formatted results with batch metadata fetch
retriever.print_results(query, results)
```

**Command Line:**
```bash
# Run demo with all 5 assignment queries
python retriever/retriever.py

# Or use pipeline script
./retriever_pipeline.sh

# View logs with latency tracking
tail -f logs/retriever.log
```

**Example Output:**
```
Using preset 'attribute_specific': α=0.4, β=0.1, γ=0.5
Expanded query: 'A person in a bright yellow raincoat' → '...yellow vibrant sunny golden'
⏱️  Retrieval Latency: 35.42 ms

Query: A person in a bright yellow raincoat
Rank 1: Image ID 34542
  Final Score: 0.2565
  Stream Scores: G=0.394, V=0.397, I=0.000
```

### 2. `evaluate.py`
Comprehensive evaluation and comparison against vanilla CLIP baseline.

**Key Features:**
- Side-by-side comparison (Triple-Stream vs Vanilla CLIP)
- Manual relevance judgments
- Ablation studies
- Visual results with image previews

**Usage:**
```bash
cd retriever
python evaluate.py
```

## API Reference

### TripleStreamRetriever

#### `dynamic_search(query, alpha=0.33, beta=0.33, gamma=0.33, top_k=10, expand=True, preset=None, filters=None, score_method='relative')`

Execute dynamic search with query-time weighting and latency tracking.

**Parameters:**
- `query` (str): Natural language search query
- `alpha` (float): Weight for grounded layer [0-1] (attributes, colors)
- `beta` (float): Weight for vibe layer [0-1] (context, scene, style)
- `gamma` (float): Weight for visual layer [0-1] (visual similarity)
- `top_k` (int): Number of results to return (default: 10)
- `expand` (bool): Enable query expansion with synonyms (default: True)
- `preset` (str): Use predefined weight preset (overrides α, β, γ)
- `filters` (dict): Optional filters:
  - `'min_garments'`: Minimum number of garments
  - `'max_garments'`: Maximum number of garments
- `score_method` (str): Normalization method - 'relative' (default), 'exponential', 'inverse'

**Returns:**
- List of `(image_id, final_score, individual_scores_dict)` tuples
- Logs latency: `⏱️ Retrieval Latency: XX.XX ms`

#### `rerank_by_color(results, query, boost_factor=1.3)`

Re-rank results by boosting color keyword matches.

**Parameters:**
- `results`: Output from `dynamic_search()`
- `query` (str): Original query string
- `boost_factor` (float): Multiplier for color matches (default: 1.3)

**Returns:**
- Re-ranked list of results

#### `get_batch_metadata(image_ids)`

Fetch metadata for multiple images in parallel (10-30x faster than sequential).

**Parameters:**
- `image_ids` (list): List of image IDs

**Returns:**
- Dictionary: `{img_id: {grounded_text, vibe_text, image_path, num_garments, ...}}`

#### `vanilla_clip_search(query, top_k=10)`

Baseline CLIP search (visual layer only).

**Parameters:**
- `query` (str): Search query
- `top_k` (int): Number of results

**Returns:**
- List of image IDs

#### `get_image_metadata(image_id)`

Retrieve full metadata for an image.

**Returns:**
- Dictionary with grounded text, vibe text, and image path

## Evaluation Queries (Assignment)

The system is designed to handle these specific query types:

1. **Attribute Specific**: "A person in a bright yellow raincoat"
2. **Contextual/Place**: "Professional business attire inside a modern office"
3. **Complex Semantic**: "Someone wearing a blue shirt sitting on a park bench"
4. **Style Inference**: "Casual weekend outfit for a city walk"
5. **Compositional**: "A red tie and a white shirt in a formal setting"

## Configuration

Weight presets and query expansion rules are configured in `shared/config.yaml`:

```yaml
weight_presets:
  attribute_specific:
    alpha: 0.4  # Attributes
    beta: 0.1   # Minimal vibe
    gamma: 0.5  # Color important

expansion_rules:
  weekend: "relaxed leisure street casual"
  bright yellow: "yellow vibrant sunny golden"
  red: "crimson scarlet burgundy"
```

## Performance (Measured on DGX Server)

**Latency:**
- First query: ~200-250ms (CLIP model warmup)
- Average query: ~30-40ms (after warmup)
- Fastest query: ~25ms

**Scale:**
- Images indexed: 45,623
- Collections: 3 (grounded, vibe, visual)
- Total vectors: 136,869

**Optimizations:**
- ✅ Batch metadata retrieval (10-30x faster)
- ✅ Efficient score normalization (relative method)
- ✅ Query expansion caching
- ✅ ChromaDB HNSW indexing (O(log n) search)

**Scalability**: Designed for millions of images with no architecture changes
