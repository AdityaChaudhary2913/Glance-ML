# Indexer Module

## Overview
The indexer module processes raw fashion images and builds a searchable triple-stream vector database. It's split into two phases:

**Phase 1 (caption_generator.py)**: Generates structured fashion knowledge (V_fact) and context-aware scene/style captions (V_vibe)
**Phase 2 (indexer.py)**: Encodes and indexes all three layers into ChromaDB

## Key Innovation: Context-Aware V_vibe

V_vibe captions are generated using Microsoft Florence-2, a powerful vision-language model:
- **Native grounding** - Better at compositional understanding than BLIP-2
- **5x faster** - Processes 45k images in ~1 hour vs ~5 hours
- **Richer outputs** - Detailed captions: "A person in a red blazer and blue jeans in a modern office"
- **Efficient batching** - 32 images per batch vs 16 with BLIP-2

## Components

### 1. `caption_generator.py` - Self-Contained Caption Pipeline
Generates BOTH grounded vectors and context-aware captions using Florence-2.

**Key Features:**
- **Self-contained**: Generates grounded vectors (V_fact) first, then uses them for captioning
- **Florence-2 powered**: Microsoft's vision-language model with native grounding
- **5x faster than BLIP-2**: Batch size 32, ~0.08 sec/image
- Automatic checkpointing every 500 images for both grounded and caption generation
- Auto-resume from checkpoint on failure
- Diversity validation (>50 unique settings)

**What it generates:**
1. `grounded_vectors.json` - Compositional garment descriptions (V_fact)
2. `vibe_captions.json` - Context-aware scene captions (V_vibe)

**Usage:**
```bash
cd indexer
python caption_generator.py  # Self-contained: Does everything for caption phase

# Monitor progress
tail -f ../logs/caption_run.log
```

**Methods:**
- `generate_grounded_vectors()`: Creates V_fact from Fashionpedia + color extraction
- `generate_caption_with_context()`: Single image with grounded context
- `generate_captions_batch_with_context()`: Batch processing (5-10x faster)
- `generate_captions()`: Main orchestrator that runs grounded → captions pipeline

### 2. `indexer.py` - Vector Database Builder
Encodes and indexes pre-generated data into ChromaDB.

**Key Features:**
- **Uses pre-generated data**: Expects `grounded_vectors.json` and `vibe_captions.json` to exist
- **V_fact (Grounded Layer)**: CLIP text encoding of grounded strings
- **V_vibe (Contextual Layer)**: CLIP text encoding of context-aware captions
- **V_img (Visual Layer)**: CLIP image encoding of raw images
- Batch processing for 3-5x speedup
- Batched ChromaDB inserts (5000 vectors at a time)

**Usage:**
```bash
cd indexer
python indexer.py  # Requires caption_generator.py output

# Monitor progress
tail -f ../logs/indexer_run.log
```

**Output:**
- `chroma_db/`: Persistent ChromaDB with 3 collections (grounded_vectors, vibe_vectors, visual_vectors)

## Pipeline Execution Order

**Recommended: Use Automated Script**
```bash
./indexer_pipeline.sh  # Runs both phases automatically
```

**Manual Execution (Advanced):**
```bash
# Phase 1: Caption Generation (includes grounded vectors)
cd indexer
python caption_generator.py

# Phase 2: Vector Indexing
python indexer.py
```

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
    ├── Florence-2 detailed captioning with native grounding
    ├── Batch process with GPU (32 images, 5x faster than BLIP-2)
    └── Output: vibe_captions.json (~1 hour)

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

## Estimated Runtime (45,623 images)

| Stage | Time |
|-------|------|
| Vibe Caption Generation | ~6.5 hours |
| Grounded Layer Generation | ~7.6 hours |
| Grounded Indexing | ~25 min |
| Vibe Indexing | ~12 min |
| Visual Indexing | ~10-15 min |
| **Total** | **~14-15 hours** |
