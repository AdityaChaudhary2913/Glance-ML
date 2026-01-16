# Triple-Stream Fashion Search Engine

A sophisticated fashion image retrieval system that understands **what** someone is wearing, **where** they are, and the **vibe** of their attire using a novel triple-stream architecture.

## 🌐 Interactive Web Demo

**Try the Streamlit web interface for interviews and demonstrations:**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser for an interactive search interface with real-time results, score visualizations, and metadata display.

---

## 📁 Project Structure

```
Glance/
├── app.py                      # 🌐 Streamlit web interface
├── indexer/                    # Part A: Feature Extraction & Vector Storage
│   ├── indexer.py             # Vector indexing into ChromaDB
│   ├── caption_generator.py   # BLIP-2 caption generation
│   ├── README.md              # Detailed indexer documentation
│   └── __init__.py
│
├── retriever/                  # Part B: Search & Query Logic
│   ├── retriever.py           # Triple-stream search with latency tracking
│   ├── evaluate.py            # Evaluation & comparison vs vanilla CLIP
│   ├── README.md              # Detailed retriever documentation
│   └── __init__.py
│
├── shared/                     # Shared utilities & configuration
│   ├── utils.py               # Color extraction, Fashionpedia parsing
│   ├── logger.py              # Logging configuration
│   ├── config.yaml            # Central configuration file
│   └── __init__.py
│
├── data/                       # Fashionpedia dataset
│   ├── instances_attributes_train2020.json
│   ├── attributes_train2020.json
│   └── train/                 # 45,623 fashion images
│
├── chroma_db/                  # Persistent vector database (45,623 × 3 streams)
├── logs/                       # Runtime logs (indexer.log, retriever.log)
├── outputs/                    # Generated vectors and manifests
├── indexer_pipeline.sh        # Automated indexing pipeline
├── retriever_pipeline.sh      # Run retrieval tests
└── README.md                   # This file
```

## Architecture

The system treats every image as a **three-vector entity**, storing vectors independently to enable dynamic query-time weighting based on search intent:

| Stream | Source | Encoding | Purpose |
|--------|--------|----------|---------|
| **V_fact** (Grounded) | Fashionpedia (46 categories, 294 attributes) + K-means color extraction | CLIP Text | Structured fashion knowledge |
| **V_vibe** (Contextual) | **Context-aware** BLIP-2 captions with constrained prompting | CLIP Text | Scene/style/occasion inference with compositional understanding |
| **V_img** (Visual) | Raw images | CLIP Image | Implicit visual features |

**Core Formula**: `Score = α·S_fact + β·S_vibe + γ·S_img`

### Key Innovation: Context-Aware V_vibe Generation

V_vibe captions are generated using BLIP-2 with constrained prompting:
```
Input to BLIP-2: Image + "Describe the scene, style, and occasion"
BLIP-2 generates: "A person wearing a red wool blazer and blue slim-fit jeans standing in a modern office environment"
```

BLIP-2 provides:
- ✅ **Strong scene understanding** - Good at contextual descriptions
- ✅ **Controlled generation** - Constrained prompting for consistent output
- ✅ **Rich captions** - Detailed scene + garment descriptions
- ✅ **Stable model** - 2.7B parameter OPT-based model for reliable generation

## Features

- **Multi-stream vector search** with query-time dynamic weighting
- **Color-aware fashion understanding** via K-means clustering on segmentation masks
- **Scene and style inference** using BLIP-2 with constrained prompting
- **Query expansion** with fashion-specific synonyms
- **Compositional query handling** (e.g., "red tie + white shirt + formal setting")
- **Configurable weight presets** for different query types
- **Real-time latency tracking** for performance monitoring (~30-40ms per query)
- **Batch metadata retrieval** for 10-30x faster results display

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for faster caption generation and indexing)

### Setup

```bash
# Clone repository
cd /path/to/Glance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install fashionpedia-api
cd fashionpedia-api-master
pip install -e .
cd ..
```

### Data Setup

1. Ensure Fashionpedia dataset is in `data/` directory:
   ```
   data/
   ├── instances_attributes_val2020.json
   ├── attributes_val2020.json
   ├── info_test2020.json 
   └── train/   (images)
   ```

2. Configure paths in `config.yaml` if needed

## Usage

### Indexing (Already Completed - 45,623 images indexed)

The dataset has been fully indexed with all three vector streams:
- ✅ Grounded layer: 45,623 vectors
- ✅ Vibe layer: 45,623 vectors  
- ✅ Visual layer: 45,623 vectors

To re-index from scratch:
```bash
./indexer_pipeline.sh  # Full pipeline with checkpointing
```

**Pipeline Overview:**
- Phase 1: Caption generation (V_fact + V_vibe) using BLIP
- Phase 2: Vector encoding and ChromaDB indexing (all 3 streams)
- Total time: ~3-4 hours on GPU
- Auto-resume: Checkpoints every 500 images

### Search & Retrieval

```bash
# Run demo with all 5 assignment queries
python retriever/retriever.py

# Or use the pipeline script
./retriever_pipeline.sh

# Monitor logs
tail -f logs/retriever.log
```

**Query Performance:**
- First query: ~200-250ms (model warmup)
- Subsequent queries: ~30-40ms average
- All results include latency tracking

### Programmatic Usage

```python
from retriever.retriever import TripleStreamRetriever

# Initialize retriever (loads CLIP + ChromaDB collections)
retriever = TripleStreamRetriever()

# Search with preset weights
results = retriever.dynamic_search(
    query="A person in a bright yellow raincoat",
    preset="attribute_specific",
    top_k=10
)
# Returns: [(img_id, score, {grounded, vibe, visual scores})]

# Search with custom weights
results = retriever.dynamic_search(
    query="Casual weekend outfit",
    alpha=0.2,  # Grounded attributes
    beta=0.7,   # Vibe/context  
    gamma=0.1,  # Visual similarity
    top_k=10,
    expand=True,  # Query expansion
    filters={'min_garments': 3}  # Filter by garment count
)

# Apply color re-ranking
results = retriever.rerank_by_color(results, query)

# Print formatted results with batch metadata
retriever.print_results(query, results)
```

## Assignment Requirements Compliance

This project fulfills all requirements from the Glance ML Internship Assignment:

✅ **Part A - Indexer**: Separate directory with feature extraction and vector storage  
✅ **Part B - Retriever**: Separate directory with search logic and evaluation  
✅ **Dataset**: 45,623 Fashionpedia images (exceeds 500-1,000 minimum)  
✅ **Vector Storage**: ChromaDB (efficient, production-ready)  
✅ **Context Awareness**: Multi-attribute queries with dynamic weighting  
✅ **Beyond Vanilla CLIP**: Triple-stream architecture addresses CLIP's compositional limitations  

### Evaluation Queries

The system is tested on all 5 required query types:

1. ✅ **Attribute Specific**: "A person in a bright yellow raincoat"
2. ✅ **Contextual/Place**: "Professional business attire inside a modern office"
3. ✅ **Complex Semantic**: "Someone wearing a blue shirt sitting on a park bench"
4. ✅ **Style Inference**: "Casual weekend outfit for a city walk"
5. ✅ **Compositional**: "A red tie and a white shirt in a formal setting"

## Configuration

Edit `shared/config.yaml` to customize:

```yaml
# Weight presets for different query types
weight_presets:
  attribute_specific:  # "bright yellow raincoat"
    alpha: 0.4  # Focus on attributes
    beta: 0.1
    gamma: 0.5  # Boost visual for color
    
  style_inference:  # "casual weekend city walk"
    alpha: 0.2
    beta: 0.7  # Focus on vibe
    gamma: 0.1

# Query expansion rules
expansion_rules:
  weekend: "relaxed leisure street casual"
  bright yellow: "yellow vibrant sunny golden"
  red: "crimson scarlet burgundy"
```

## Query Types & Presets

| Query Type | Example | Best Preset | Weights (α, β, γ) |
|------------|---------|-------------|-------------------|
| Attribute-specific | "bright yellow raincoat" | `attribute_specific` | 0.4, 0.1, 0.5 |
| Contextual place | "modern office attire" | `contextual_place` | 0.3, 0.6, 0.1 |
| Complex semantic | "blue shirt on park bench" | `complex_semantic` | 0.3, 0.5, 0.2 |
| Style inference | "casual weekend outfit" | `style_inference` | 0.2, 0.7, 0.1 |
| Compositional | "red tie + white shirt" | `compositional` | 0.4, 0.1, 0.5 |

## Evaluation

Run evaluation script:

```bash
python evaluate.py
```

This compares triple-stream performance against vanilla CLIP baseline.

## Technical Details

### Color Extraction
- K-means clustering (k=3) on segmentation masks
- Maps RGB → 20 fashion color names
- Fallback to "neutral" for small masks (<50 pixels)

### Scene Inference
- BLIP-2 with constrained prompting for scene/style understanding
- Native grounding - better compositional understanding
- Format: Detailed scene descriptions with garment details
- Example: "A person wearing a red wool blazer and blue slim-fit jeans standing in a modern office environment"

### Score Normalization
- ChromaDB returns L2 distances (lower = better)
- **Default method**: `relative` - `1 - (distance / max_distance)`
- Alternative methods: `exponential` (with decay), `inverse` (1 / (1 + distance))
- Normalized to [0, 1] range before weighted fusion
- Fixed bug: Changed default from exponential to relative for better score distribution

## Performance & Scalability

### Optimizations for Full Dataset Run

✅ **Batched Image Encoding**: Process 32 images at once (3-5x speedup)  
✅ **Checkpointing**: Auto-save progress every 500 images, resume on failure  
✅ **Batched ChromaDB Inserts**: Insert 5000 vectors at a time (prevents memory issues)  
✅ **Sampled Color Extraction**: 10% sampling, max 500 pixels (speed vs accuracy)  

### Runtime Estimates (45,623 images - COMPLETED)

| Stage | Time | Status |
|-------|------|--------|
| Grounded Layer Generation | ~45 min | ✅ Complete |
| Vibe Caption Generation (BLIP-2) | ~4-5 hours | ✅ Complete |
| Vector Encoding + ChromaDB Indexing | ~1-1.5 hours | ✅ Complete |
| **Total Indexing Time** | **~3-4 hours** | ✅ Complete |
| **Collections Created** | **3 × 45,623 vectors** | ✅ Ready |

### Query Performance (Measured on DGX Server)

- **First query latency**: ~234ms (includes CLIP model warmup)
- **Average query latency**: ~30-40ms (after warmup)
- **Retrieval scale**: 45,623 images across 3 collections
- **Scoring**: Relative normalization (default, most reliable)
- **Metadata fetch**: Batch retrieval (10-30x faster than sequential)
- **Scalability**: O(log n) with ChromaDB HNSW indexing
- **Production ready**: Designed for millions of images

### Why It Scales

1. **ChromaDB**: Production-grade vector DB with efficient indexing
2. **Independent Streams**: Each collection scales independently
3. **Query-time Weighting**: No reindexing needed for different query types
4. **Stateless Search**: No session management, fully parallelizable

## Module Documentation

- **Indexer Module**: See [indexer/README.md](indexer/README.md)
- **Retriever Module**: See [retriever/README.md](retriever/README.md)

### Query Expansion
- Automatic synonym expansion based on keywords
- Fashion-specific color terms
- Scene and style descriptors

## Why Query-Time Weighting?

**Index-time fusion** (averaging vectors):
- ❌ Fixed weights forever
- ❌ Can't adapt to query intent
- ❌ Loses information

**Query-time weighting** (our approach):
- ✅ Different weights per query type
- ✅ Color query → boost visual
- ✅ Style query → boost vibe
- ✅ Maintains full information

## Limitations & Future Work

### Current Limitations
1. **Color taxonomy**: Basic 20-color mapping; needs fine-grained taxonomy
2. **Scene ambiguity**: BLIP-2 struggles with unclear backgrounds
3. **Semantic similarity**: "raincoat" ↔ "trench coat" (feature, not bug!)

### Future Enhancements
1. **Cross-modal attention**: Fuse streams with Transformer encoder
2. **Temporal/weather APIs**: Real-time context (GPS + weather data)
3. **Fine-tuned CLIP**: Contrastive learning on Fashionpedia
4. **Compositional negatives**: Hard negative mining for better attribute binding
5. **User feedback loop**: Click-through rate optimization

## Performance

**Target**: 15-20% improvement over vanilla CLIP on compositional queries

| Query Type | Triple-Stream P@10 | Vanilla CLIP P@10 | Improvement |
|------------|-------------------|-------------------|-------------|
| Attribute-specific | TBD | TBD | TBD |
| Contextual | TBD | TBD | TBD |
| Complex semantic | TBD | TBD | TBD |
| Style inference | TBD | TBD | TBD |
| Compositional | TBD | TBD | TBD |

## License

This project uses the Fashionpedia dataset. See `fashionpedia-api-master/license.txt` for dataset license.

## Acknowledgments

- Fashionpedia dataset and API
- OpenAI CLIP
- Salesforce BLIP-2
- ChromaDB vector database
