# Triple-Stream Fashion Search Engine

A sophisticated fashion image retrieval system that understands **what** someone is wearing, **where** they are, and the **vibe** of their attire using a novel triple-stream architecture.

## Architecture

The system treats every image as a **three-vector entity**, storing vectors independently to enable dynamic query-time weighting based on search intent:

| Stream | Source | Encoding | Purpose |
|--------|--------|----------|---------|
| **V_fact** (Grounded) | Fashionpedia (46 categories, 294 attributes) + K-means color extraction | CLIP Text | Structured fashion knowledge |
| **V_vibe** (Contextual) | BLIP-2 constrained captions | CLIP Text | Scene/style/occasion inference |
| **V_img** (Visual) | Raw images | CLIP Image | Implicit visual features |

**Core Formula**: `Score = α·S_fact + β·S_vibe + γ·S_img`

## Features

- **Multi-stream vector search** with query-time dynamic weighting
- **Color-aware fashion understanding** via segmentation-based extraction
- **Scene and style inference** using BLIP-2 constrained prompting
- **Query expansion** with fashion-specific synonyms
- **Compositional query handling** (e.g., "red tie + white shirt + formal setting")
- **Configurable weight presets** for different query types

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for BLIP-2)

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
   ├── train/  (images)
   └── test/   (images)
   ```

2. Configure paths in `config.yaml` if needed

## Usage

### 1. Generate Vibe Captions

```bash
python caption_generator.py
```

This generates scene/style captions using BLIP-2 for 2,500 images and saves to `vibe_captions.json`.

**Expected output:**
- `vibe_captions.json`: Mapping of image_id → caption
- Diversity check: Should have >50 unique settings

### 2. Build Vector Index

```bash
python indexer.py
```

This creates three ChromaDB collections:
- **grounded_vectors**: Fashionpedia annotations + extracted colors
- **vibe_vectors**: BLIP-2 scene/style captions
- **visual_vectors**: CLIP image embeddings

**Expected output:**
- `chroma_db/`: Persistent vector database
- `grounded_data.json`: Structured fashion descriptions

### 3. Search Images

```bash
python retriever.py
```

Or use programmatically:

```python
from retriever import TripleStreamRetriever

retriever = TripleStreamRetriever()

# Search with preset weights
results = retriever.dynamic_search(
    query="A person in a bright yellow raincoat",
    preset="attribute_specific"
)

# Search with custom weights
results = retriever.dynamic_search(
    query="Casual weekend outfit",
    alpha=0.2,  # Grounded
    beta=0.7,   # Vibe
    gamma=0.1,  # Visual
    top_k=10
)

# Print results
retriever.print_results(query, results)
```

## Configuration

Edit `config.yaml` to customize:

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

## Project Structure

```
Glance/
├── config.yaml              # Configuration file
├── requirements.txt         # Dependencies
├── utils.py                 # Helper functions
├── caption_generator.py     # BLIP-2 vibe caption generation
├── indexer.py              # Multi-stream vectorization
├── retriever.py            # Dynamic search engine
├── evaluate.py             # Evaluation & metrics
├── README.md               # This file
├── data/                   # Fashionpedia dataset
├── chroma_db/              # Vector database (generated)
├── vibe_captions.json      # Generated captions
└── grounded_data.json      # Generated descriptions
```

## Technical Details

### Color Extraction
- K-means clustering (k=3) on segmentation masks
- Maps RGB → 20 fashion color names
- Fallback to "neutral" for small masks (<50 pixels)

### Scene Inference
- BLIP-2 with constrained prompt
- Format: "[setting], [vibe/occasion]"
- Example: "Indoor office, professional attire"

### Score Normalization
- ChromaDB returns L2 distances (lower = better)
- Converted to similarities: `similarity = 1 - (distance / max_distance)`
- Normalized to [0, 1] range before weighted fusion

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

*(Run evaluation to populate)*

## Citation

```bibtex
@misc{fashionpedia2020,
  title={Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset},
  author={Jia, Menglin and Shi, Mengyun and Sirotenko, Mikhail and others},
  year={2020}
}
```

## License

This project uses the Fashionpedia dataset. See `fashionpedia-api-master/license.txt` for dataset license.

## Acknowledgments

- Fashionpedia dataset and API
- OpenAI CLIP
- Salesforce BLIP-2
- ChromaDB vector database
