# Indexer Module

## Overview
The indexer module processes raw fashion images and builds a searchable triple-stream vector database. It extracts structured fashion knowledge (V_fact), generates scene/style captions (V_vibe), and encodes raw visual features (V_img).

## Components

### 1. `caption_generator.py`
Generates vibe captions using BLIP-2 for understanding scene context and style.

**Key Features:**
- BLIP-2 constrained prompting for consistent captions
- Automatic checkpointing every 500 images
- Auto-resume from checkpoint on failure
- Diversity validation (>50 unique settings)

**Usage:**
```bash
cd indexer
python caption_generator.py
```

**Output:**
- `vibe_captions.json`: Image ID → caption mapping
- `vibe_captions_checkpoint.json`: (temporary, removed on completion)

### 2. `indexer.py`
Builds three independent vector collections in ChromaDB.

**Key Features:**
- **V_fact (Grounded Layer)**: Fashionpedia annotations + K-means color extraction
- **V_vibe (Contextual Layer)**: BLIP-2 captions encoded with CLIP Text
- **V_img (Visual Layer)**: Raw images encoded with CLIP Image
- Batch processing for 3-5x speedup
- Automatic checkpointing for long runs
- Batched ChromaDB inserts (5000 vectors at a time)

**Usage:**
```bash
cd indexer
python indexer.py
```

**Output:**
- `chroma_db/`: Persistent ChromaDB with 3 collections
- `grounded_data.json`: Structured fashion descriptions
- `grounded_data_checkpoint.json`: (temporary, removed on completion)

## Pipeline Execution Order

1. **Generate Vibe Captions First** (slowest: ~6.5 hours for 45K images)
   ```bash
   nohup python caption_generator.py > caption_run.log 2>&1 &
   ```

2. **Build Vector Index** (after captions are done)
   ```bash
   nohup python indexer.py > indexer_run.log 2>&1 &
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
