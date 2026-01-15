"""
Indexer: Multi-Stream Vectorization Pipeline
Generates and stores V_fact, V_vibe, and V_img vectors in ChromaDB
"""

import sys
import os
sys.path.insert(0, '/workspace/fashionpedia-api-master')
# Add parent directory to path for shared modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import yaml
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import chromadb
from chromadb.config import Settings

from shared.utils import (
    load_fashionpedia_data,
    get_image_annotations,
    extract_dominant_colors,
    build_grounded_string,
    ensure_dir,
    save_json,
    load_json
)
from shared.logger import indexer_logger as logger


class MultiStreamIndexer:
    """
    Builds three independent vector collections for fashion image search
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize indexer with configuration
        
        Args:
            config_path: Path to config.yaml file (None = auto-detect)
        """
        # Load configuration
        if config_path is None:
            # Get path relative to project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            config_path = os.path.join(project_root, 'shared', 'config.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize CLIP model
        logger.info(f"Loading CLIP model: {self.config['models']['clip_model']}")
        self.clip_model = SentenceTransformer(self.config['models']['clip_model'])
        
        # Get project root for data paths
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Initialize ChromaDB client
        persist_dir = os.path.join(self.project_root, self.config['chromadb']['persist_directory'])
        ensure_dir(persist_dir)
        
        logger.info(f"Initializing ChromaDB at {persist_dir}")
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Create collections (delete if exists unless incremental mode)
        collection_names = self.config['chromadb']['collections']
        self.collections = {}
        incremental = self.config.get('indexing', {}).get('incremental_mode', False)
        
        for key, name in collection_names.items():
            if not incremental:
                try:
                    self.client.delete_collection(name)
                    logger.info(f"Deleted existing collection: {name}")
                except:
                    pass
                self.collections[key] = self.client.create_collection(name)
                logger.info(f"Created fresh collection: {name}")
            else:
                try:
                    self.collections[key] = self.client.get_collection(name)
                    count = self.collections[key].count()
                    logger.info(f"Loaded existing collection: {name} ({count} vectors)")
                except:
                    self.collections[key] = self.client.create_collection(name)
                    logger.info(f"Created new collection: {name}")
        
        # Load Fashionpedia data with absolute paths
        annotations_path = os.path.join(self.project_root, self.config['data']['annotations_path'])
        attributes_path = os.path.join(self.project_root, self.config['data']['attributes_path'])
        self.fp, self.attr_data = load_fashionpedia_data(annotations_path, attributes_path)
        
    def extract_colors_for_image(
        self,
        image_id: int,
        annotations: List[Dict]
    ) -> List[List[str]]:
        """
        Extract dominant colors for all garments in an image
        
        Args:
            image_id: Image ID
            annotations: List of annotations for this image
            
        Returns:
            List of color lists for each annotation
        """
        # Get image info
        img_info = self.fp.loadImgs([image_id])[0]
        img_filename = img_info['file_name']
        img_path = os.path.join(self.project_root, self.config['data']['images_dir'], img_filename)
        
        colors_list = []
        for ann in annotations:
            segmentation = ann.get('segmentation', [])
            is_crowd = ann.get('iscrowd', 0)
            
            colors = extract_dominant_colors(
                image_path=img_path,
                segmentation=segmentation,
                is_crowd=is_crowd,
                k=self.config['color_extraction']['kmeans_clusters'],
                min_pixels=self.config['color_extraction']['min_pixels']
            )
            colors_list.append(colors)
        
        return colors_list
    
    def generate_grounded_vectors(
        self,
        image_ids: List[int],
        output_path: str = "grounded_data.json",
        checkpoint_interval: int = 500,
        resume: bool = True
    ) -> Dict:
        """
        Generate V_fact: Grounded layer vectors from Fashionpedia + colors
        WITH CHECKPOINTING for long runs
        
        Args:
            image_ids: List of image IDs to process
            output_path: Path to save grounded strings
            checkpoint_interval: Save progress every N images
            resume: Whether to resume from checkpoint
            
        Returns:
            Dictionary mapping image_id to grounded string
        """
        logger.info("=== Generating Grounded Layer (V_fact) ===")
        logger.info(f"Checkpoint interval: every {checkpoint_interval} images")
        
        grounded_data = {}
        checkpoint_path = output_path.replace('.json', '_checkpoint.json')
        
        # Resume from checkpoint if exists
        if resume and os.path.exists(checkpoint_path):
            try:
                grounded_data = load_json(checkpoint_path)
                logger.info(f"✓ Resumed from checkpoint: {len(grounded_data)} images already done")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        # Filter out already processed images
        processed_ids = set(grounded_data.keys())
        remaining_ids = [img_id for img_id in image_ids if str(img_id) not in processed_ids]
        
        if len(remaining_ids) < len(image_ids):
            logger.info(f"Skipping {len(image_ids) - len(remaining_ids)} already processed images")
        
        processed_count = 0
        
        for img_id in tqdm(remaining_ids, desc="Processing grounded strings"):
            # Get annotations for this image
            annotations = get_image_annotations(self.fp, img_id)
            
            if not annotations:
                continue
            
            # Extract colors for all garments
            colors_list = self.extract_colors_for_image(img_id, annotations)
            
            # Build grounded string
            grounded_str = build_grounded_string(
                self.fp,
                img_id,
                annotations,
                colors_list,
                max_attributes=3
            )
            
            # Store metadata
            img_info = self.fp.loadImgs([img_id])[0]
            grounded_data[str(img_id)] = {
                'text': grounded_str,
                'image_path': os.path.join(self.project_root, self.config['data']['images_dir'], img_info['file_name']),
                'categories': [ann['category_id'] for ann in annotations],
                'colors': colors_list
            }
            
            processed_count += 1
            
            # Save checkpoint periodically
            if processed_count % checkpoint_interval == 0:
                save_json(grounded_data, checkpoint_path)
                logger.info(f"✓ Checkpoint saved: {len(grounded_data)} images")
        
        # Save final grounded data
        save_json(grounded_data, output_path)
        
        # Remove checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info(f"✓ Removed checkpoint file (full run completed)")
        
        return grounded_data
    
    def index_grounded_layer(self, grounded_data: Dict):
        """
        Encode and store grounded strings in ChromaDB with batched inserts
        
        Args:
            grounded_data: Dictionary from generate_grounded_vectors
        """
        import time
        start_time = time.time()
        
        logger.info("=== Indexing Grounded Layer ===")
        
        image_ids = []
        texts = []
        metadatas = []
        
        for img_id, data in grounded_data.items():
            image_ids.append(str(img_id))
            texts.append(data['text'])
            metadatas.append({
                'text': data['text'],
                'image_path': data['image_path'],
                'categories': json.dumps(data['categories']),
                'colors': json.dumps(data['colors'])
            })
        
        # Get batch sizes from config
        encoding_batch = self.config.get('indexing', {}).get('encoding_batch_size', 32)
        chromadb_batch = self.config.get('indexing', {}).get('chromadb_batch_size', 5000)
        
        # Encode with CLIP Text Encoder
        logger.info(f"Encoding {len(texts)} grounded strings with CLIP (batch_size={encoding_batch})...")
        encode_start = time.time()
        embeddings = self.clip_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=encoding_batch
        )
        encode_time = time.time() - encode_start
        logger.info(f"  Encoding took {encode_time:.1f}s ({len(texts)/encode_time:.1f} texts/sec)")
        
        # Store in ChromaDB in batches
        logger.info(f"Storing in ChromaDB (batch_size={chromadb_batch})...")
        insert_start = time.time()
        for i in range(0, len(image_ids), chromadb_batch):
            end_idx = min(i + chromadb_batch, len(image_ids))
            self.collections['grounded'].add(
                embeddings=embeddings[i:end_idx].tolist(),
                ids=image_ids[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            logger.info(f"  Inserted batch {i//chromadb_batch + 1}: {end_idx}/{len(image_ids)}")
        insert_time = time.time() - insert_start
        
        total_time = time.time() - start_time
        logger.info(f"✓ Indexed {len(image_ids)} grounded vectors in {total_time:.1f}s")
    
    def index_vibe_layer(self, vibe_captions_path: str = "vibe_captions.json", grounded_data: Dict = None):
        """
        Encode and store vibe captions in ChromaDB with batched inserts
        
        Args:
            vibe_captions_path: Path to vibe captions JSON file
            grounded_data: Optional grounded data for validation
        """
        import time
        start_time = time.time()
        
        logger.info("=== Indexing Vibe Layer (V_vibe) ===")
        
        # Load vibe captions
        vibe_data = load_json(vibe_captions_path)
        
        # Validate consistency with grounded layer
        if grounded_data:
            grounded_ids = set(grounded_data.keys())
            vibe_ids = set(vibe_data.keys())
            if grounded_ids != vibe_ids:
                missing_in_vibe = grounded_ids - vibe_ids
                missing_in_grounded = vibe_ids - grounded_ids
                if missing_in_vibe:
                    logger.warning(f"⚠ {len(missing_in_vibe)} images in grounded but not in vibe")
                if missing_in_grounded:
                    logger.warning(f"⚠ {len(missing_in_grounded)} images in vibe but not in grounded")
            else:
                logger.info(f"✓ Consistency check passed: {len(vibe_ids)} images match")
        
        image_ids = []
        texts = []
        metadatas = []
        
        for img_id, caption in vibe_data.items():
            image_ids.append(str(img_id))
            texts.append(caption)
            
            # Get image path
            img_info = self.fp.loadImgs([int(img_id)])[0]
            img_path = os.path.join(self.project_root, self.config['data']['images_dir'], img_info['file_name'])
            
            metadatas.append({
                'text': caption,
                'image_path': img_path
            })
        
        # Get batch sizes from config
        encoding_batch = self.config.get('indexing', {}).get('encoding_batch_size', 32)
        chromadb_batch = self.config.get('indexing', {}).get('chromadb_batch_size', 5000)
        
        # Encode with CLIP Text Encoder
        logger.info(f"Encoding {len(texts)} vibe captions with CLIP (batch_size={encoding_batch})...")
        encode_start = time.time()
        embeddings = self.clip_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=encoding_batch
        )
        encode_time = time.time() - encode_start
        logger.info(f"  Encoding took {encode_time:.1f}s ({len(texts)/encode_time:.1f} texts/sec)")
        
        # Store in ChromaDB in batches
        logger.info(f"Storing in ChromaDB (batch_size={chromadb_batch})...")
        insert_start = time.time()
        for i in range(0, len(image_ids), chromadb_batch):
            end_idx = min(i + chromadb_batch, len(image_ids))
            self.collections['vibe'].add(
                embeddings=embeddings[i:end_idx].tolist(),
                ids=image_ids[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            logger.info(f"  Inserted batch {i//chromadb_batch + 1}: {end_idx}/{len(image_ids)}")
        insert_time = time.time() - insert_start
        
        total_time = time.time() - start_time
        logger.info(f"✓ Indexed {len(image_ids)} vibe vectors in {total_time:.1f}s")
    
    def index_visual_layer(
        self,
        image_ids: List[int],
        checkpoint_interval: int = None,
        resume: bool = True
    ):
        """
        Encode and store raw images in ChromaDB with BATCHING and CHECKPOINTING
        
        Args:
            image_ids: List of image IDs to process
            checkpoint_interval: Save progress every N images (None = use config)
            resume: Whether to resume from checkpoint
        """
        import time
        start_time = time.time()
        
        logger.info("=== Indexing Visual Layer (V_img) ===")
        
        # Get config values
        batch_size = self.config.get('indexing', {}).get('encoding_batch_size', 32)
        if checkpoint_interval is None:
            checkpoint_interval = self.config.get('indexing', {}).get('visual_checkpoint_interval', 1000)
        save_failed = self.config.get('indexing', {}).get('save_failed_ids', True)
        
        logger.info(f"Batch size: {batch_size}, Checkpoint interval: {checkpoint_interval}")
        
        checkpoint_path = os.path.join(self.project_root, 'visual_index_checkpoint.json')
        failed_ids_path = os.path.join(self.project_root, 'visual_failed_ids.json')
        
        # Load checkpoint if exists
        indexed_ids = set()
        failed_ids = []
        
        if resume and os.path.exists(checkpoint_path):
            try:
                checkpoint_data = load_json(checkpoint_path)
                indexed_ids = set(checkpoint_data.get('indexed_ids', []))
                failed_ids = checkpoint_data.get('failed_ids', [])
                logger.info(f"✓ Resumed from checkpoint: {len(indexed_ids)} images already indexed")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        # Filter out already indexed images
        remaining_ids = [img_id for img_id in image_ids if img_id not in indexed_ids]
        
        if len(remaining_ids) < len(image_ids):
            logger.info(f"Skipping {len(image_ids) - len(remaining_ids)} already indexed images")
        
        total_indexed = len(indexed_ids)
        processed_since_checkpoint = 0
        
        # Process in batches
        for batch_start in tqdm(range(0, len(remaining_ids), batch_size), desc="Encoding images"):
            batch_ids = remaining_ids[batch_start:batch_start + batch_size]
            
            batch_images = []
            batch_img_ids = []
            batch_metadatas = []
            
            for img_id in batch_ids:
                # Get image path
                try:
                    img_info = self.fp.loadImgs([img_id])[0]
                    img_filename = img_info['file_name']
                    img_path = os.path.join(self.project_root, self.config['data']['images_dir'], img_filename)
                    
                    if not os.path.exists(img_path):
                        failed_ids.append(img_id)
                        continue
                    
                    # Load image
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(img)
                    batch_img_ids.append(str(img_id))
                    batch_metadatas.append({'image_path': img_path})
                    
                except Exception as e:
                    logger.error(f"Error loading image {img_id}: {e}")
                    failed_ids.append(img_id)
                    continue
            
            if not batch_images:
                continue
            
            try:
                # Batch encode with CLIP
                embeddings = self.clip_model.encode(batch_images, convert_to_numpy=True)
                
                # Store batch in ChromaDB
                self.collections['visual'].add(
                    embeddings=embeddings.tolist(),
                    ids=batch_img_ids,
                    metadatas=batch_metadatas
                )
                
                # Update tracking
                for img_id_str in batch_img_ids:
                    indexed_ids.add(int(img_id_str))
                
                total_indexed += len(batch_img_ids)
                processed_since_checkpoint += len(batch_img_ids)
                
                # Save checkpoint periodically
                if processed_since_checkpoint >= checkpoint_interval:
                    checkpoint_data = {
                        'indexed_ids': list(indexed_ids),
                        'failed_ids': failed_ids,
                        'total_indexed': total_indexed
                    }
                    save_json(checkpoint_data, checkpoint_path)
                    logger.info(f"✓ Checkpoint saved: {total_indexed} images indexed")
                    processed_since_checkpoint = 0
                
            except Exception as e:
                logger.error(f"Batch encoding failed at {batch_start}: {e}")
                failed_ids.extend([int(i) for i in batch_img_ids])
        
        # Save failed IDs to file if enabled
        if save_failed and failed_ids:
            save_json({'failed_ids': failed_ids, 'count': len(failed_ids)}, failed_ids_path)
            logger.warning(f"⚠ Saved {len(failed_ids)} failed IDs to: {failed_ids_path}")
        
        # Remove checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info(f"✓ Removed checkpoint file (indexing completed)")
        
        total_time = time.time() - start_time
        throughput = total_indexed / total_time if total_time > 0 else 0
        logger.info(f"✓ Indexed {total_indexed} visual vectors in {total_time:.1f}s ({throughput:.1f} images/sec)")
        
        if failed_ids:
            logger.warning(f"⚠ {len(failed_ids)} images failed to process")
    
    def build_index(self, num_images: int = None):
        """
        Main pipeline: Build all three vector indices from pre-generated data
        Expects grounded_vectors.json and vibe_captions.json to already exist
        (generated by caption_generator.py)
        
        Args:
            num_images: Number of images to process (None = use config)
        """
        if num_images is None:
            num_images = self.config['data']['num_images']
            if num_images == -1:
                num_images = len(self.fp.getImgIds())
        
        logger.info("\n" + "="*70)
        logger.info("INDEXER: Building Triple-Stream Vector Database")
        logger.info("="*70)
        logger.info(f"Processing {num_images} images")
        logger.info("")
        
        # Check prerequisites
        grounded_path = os.path.join(self.project_root, "grounded_vectors.json")
        vibe_path = os.path.join(self.project_root, "vibe_captions.json")
        
        if not os.path.exists(grounded_path):
            logger.error(f"❌ grounded_vectors.json not found at: {grounded_path}")
            logger.error("Please run caption generation first:")
            logger.error("  cd indexer && python caption_generator.py")
            raise FileNotFoundError(f"Required file not found: {grounded_path}")
        
        if not os.path.exists(vibe_path):
            logger.error(f"❌ vibe_captions.json not found at: {vibe_path}")
            logger.error("Please run caption generation first:")
            logger.error("  cd indexer && python caption_generator.py")
            raise FileNotFoundError(f"Required file not found: {vibe_path}")
        
        logger.info("✓ Found grounded_vectors.json")
        logger.info("✓ Found vibe_captions.json")
        logger.info("")
        
        # Get image IDs
        all_img_ids = self.fp.getImgIds()
        image_ids = all_img_ids[:num_images]
        
        # Start overall timer
        import time
        pipeline_start = time.time()
        
        # Load grounded data
        logger.info("="*60)
        logger.info("STEP 1/3: Indexing Grounded Layer (V_fact)")
        logger.info("="*60)
        grounded_data = load_json(grounded_path)
        self.index_grounded_layer(grounded_data)
        
        # Index vibe layer with validation
        logger.info("\n" + "="*60)
        logger.info("STEP 2/3: Indexing Vibe Layer (V_vibe)")
        logger.info("="*60)
        self.index_vibe_layer(vibe_path, grounded_data=grounded_data)
        
        # Index visual layer with checkpointing
        logger.info("\n" + "="*60)
        logger.info("STEP 3/3: Indexing Visual Layer (V_img)")
        logger.info("="*60)
        self.index_visual_layer(image_ids)
        
        pipeline_time = time.time() - pipeline_start
        
        logger.info("\n" + "="*70)
        logger.info("✓ INDEXING COMPLETE")
        logger.info("="*70)
        logger.info(f"Total pipeline time: {pipeline_time:.1f}s ({pipeline_time/60:.1f} minutes)")
        logger.info("ChromaDB collections created:")
        for key, collection in self.collections.items():
            count = collection.count()
            logger.info(f"  - {key}: {count} vectors")
        
        logger.info("")
        logger.info("Vector database ready at: " + os.path.join(self.project_root, self.config['chromadb']['persist_directory']))
        logger.info("")
        logger.info("Next step: Run search queries")
        logger.info("  cd retriever && python retriever.py")


def main():
    """Main execution"""
    indexer = MultiStreamIndexer()
    indexer.build_index()


if __name__ == "__main__":
    main()
