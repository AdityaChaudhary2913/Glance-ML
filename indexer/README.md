# Indexer Module

## Overview
The indexer module processes raw fashion images and builds a searchable triple-stream vector database.

**Current Status**: ✅ **COMPLETED** - All 45,623 images indexed across 3 streams

**Architecture**: Two-phase pipeline
- **Phase 1 (`caption_generator.py`)**: Generate V_fact (grounded) + V_vibe (captions) using BLIP-2
- **Phase 2 (`indexer.py`)**: Encode all layers with CLIP and index into ChromaDB

**Total Collections**: 3 (grounded_vectors, vibe_vectors, visual_vectors) × 45,623 = 136,869 vectors

## Key Innovation: Context-Aware V_vibe with BLIP-2

V_vibe captions are generated using BLIP-2 with constrained prompting:
- **Strong scene understanding** - Excellent at contextual and compositional descriptions
- **Controlled generation** - Uses custom prompts: "Describe the scene, style, and occasion"
- **Rich captions** - "A person wearing a red blazer and blue slim jeans in a modern office"
- **Stable architecture** - 2.7B parameter OPT-based model (blip2-opt-2.7b)
- **Reliable output** - Consistent caption quality across diverse fashion images

## Components

### 1. `caption_generator.py` - Grounded + Caption Generation Pipeline

**Purpose**: Generates both V_fact (grounded attributes) and V_vibe (scene captions)

**Key Features:**
- Self-contained dual-phase generator
- BLIP-2 powered caption generation with constrained prompting
- K-means color extraction from Fashionpedia segmentation masks
- Automatic checkpointing every 500 images
- Auto-resume capability on interruption

**Output Files:**
1. `outputs/grounded_vectors.json` - Compositional garment descriptions
2. `outputs/vibe_captions.json` - Context-aware scene captions  
3. `outputs/grounded_layer_manifest.json` - Layer metadata
4. `outputs/vibe_layer_manifest.json` - Layer metadata

**Status**: ✅ Completed for all 45,623 images

**Usage:**
```bash
python caption_generator.py
```

### 2. `indexer.py` - ChromaDB Vector Database Builder

**Purpose**: Encodes pre-generated data and indexes into ChromaDB collections

**Key Features:**
- Loads from `outputs/grounded_vectors.json` and `outputs/vibe_captions.json`
- CLIP text encoding for V_fact and V_vibe
- CLIP image encoding for V_img (visual layer)
- Batch processing for 5-10x speedup
- Batched ChromaDB inserts (5000 vectors per batch)
- Persistence to `chroma_db/` directory

**Output Collections:**
- `grounded_vectors` - 45,623 text embeddings
- `vibe_vectors` - 45,623 text embeddings
- `visual_vectors` - 45,623 image embeddings

**Status**: ✅ Completed, collections ready for retrieval

**Usage:**
```bash
python indexer.py  # Requires caption_generator output
```

## Pipeline Execution

**Automated (Recommended)**:
```bash
./indexer_pipeline.sh  # Runs both phases automatically
```

**Manual (For debugging)**:
```bash
# Phase 1: Generate grounded + captions
cd indexer
python caption_generator.py

# Phase 2: Encode and index
python indexer.py
cd ..
```

**Current Status**: Both phases completed, system ready for retrieval

### Pipeline Flow

```
Phase 1: caption_generator.py
├── Step 1: Generate grounded vectors (V_fact)
│   ├── Extract Fashionpedia annotations
│   ├── K-means color extraction from segmentation masks
│   ├── Build compositional descriptions
│   └── Output: grounded_vectors.json (~2-3 hours)
│
└── Step 2: Generate context-aware captions (V_vibe)
    ├── Load grounded vectors from Step 1
    ├── BLIP-2 detailed captioning with constrained prompting
    ├── Batch process with GPU for scene and style understanding
    └── Output: vibe_captions.json (~4-5 hours)

Phase 2: indexer.py
├── Step 1: Index grounded layer (V_fact)
│   ├── Load grounded_vectors.json
│   ├── CLIP text encode
│   └── Store in ChromaDB
│
├── Step 2: Index vibe layer (V_vibe)
│   ├── Load vibe_captions.json
│   ├── CLIP text encode
│   └── Store in ChromaDB
│
└── Step 3: Index visual layer (V_img)
    ├── Load raw images
    ├── CLIP image encode (batched)
    └── Store in ChromaDB
```

## Checkpoint/Resume

Both scripts support automatic checkpointing:
- If a script crashes or is interrupted, simply re-run the same command
- It will automatically resume from the last checkpoint
- Checkpoints are saved every 500 images
- Checkpoint files are deleted after successful completion

## Configuration

Edit `shared/config.yaml` to customize:
- Model selection (CLIP, BLIP-2)
- Data paths
- Color extraction parameters
- ChromaDB settings

## Performance Optimizations

- ✅ Batched CLIP encoding (32 images at once)
- ✅ Sampled color extraction (10%, max 500 pixels)
- ✅ Checkpointing for crash recovery
- ✅ Batched ChromaDB inserts (5000 vectors)

## Completed Runtime (45,623 images)

**Actual measured times from logs (2026-01-16):**

| Stage | Time | Throughput | Status |
|-------|------|------------|--------|
| Grounded Layer Generation (STEP 1) | **249 min (4h 9m)** | 3.05 images/sec | ✅ |
| Vibe Caption Generation - BLIP-2 (STEP 2) | **199 min (3h 19m)** | 3.82 images/sec | ✅ |
| **Total Caption Generation** | **448 min (7.5 hours)** | 1.70 images/sec | ✅ |
| Vector Encoding + ChromaDB Indexing | **17 min** | 135.6 vectors/sec | ✅ |
| **Total Pipeline Time** | **465 min (7.75 hours)** | - | ✅ |

**Hardware**: NVIDIA DGX Server with GPU acceleration

**Output**: 136,869 vectors indexed across 3 collections:
- `grounded_vectors`: 45,623 vectors (Fashionpedia + colors)
- `vibe_vectors`: 45,623 vectors (BLIP-2 scene captions)
- `visual_vectors`: 45,623 vectors (CLIP image embeddings)

**Performance Notes**:
- BLIP-2 captioning is the bottleneck (~3h 19m for rich contextual descriptions)
- Grounded generation includes Fashionpedia parsing + K-means color extraction
- Vector encoding highly optimized with batched CLIP encoding
- ChromaDB indexing extremely fast with batched inserts (5000 vectors/batch)
