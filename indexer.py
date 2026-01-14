"""
Indexer: Multi-Stream Vectorization Pipeline
Generates and stores V_fact, V_vibe, and V_img vectors in ChromaDB
"""

import sys
sys.path.insert(0, '/workspace/fashionpedia-api-master')

import os
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

from utils import (
    load_fashionpedia_data,
    get_image_annotations,
    extract_dominant_colors,
    build_grounded_string,
    ensure_dir,
    save_json,
    load_json
)
from logger import indexer_logger as logger


class MultiStreamIndexer:
    """
    Builds three independent vector collections for fashion image search
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize indexer with configuration
        
        Args:
            config_path: Path to config.yaml file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize CLIP model
        logger.info(f"Loading CLIP model: {self.config['models']['clip_model']}")
        self.clip_model = SentenceTransformer(self.config['models']['clip_model'])
        
        # Initialize ChromaDB client
        persist_dir = self.config['chromadb']['persist_directory']
        ensure_dir(persist_dir)
        
        logger.info(f"Initializing ChromaDB at {persist_dir}")
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Create collections (delete if exists for fresh start)
        collection_names = self.config['chromadb']['collections']
        self.collections = {}
        
        for key, name in collection_names.items():
            try:
                self.client.delete_collection(name)
            except:
                pass
            self.collections[key] = self.client.create_collection(name)
            logger.info(f"Created collection: {name}")
        
        # Load Fashionpedia data
        self.fp, self.attr_data = load_fashionpedia_data(
            self.config['data']['annotations_path'],
            self.config['data']['attributes_path']
        )
        
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
        img_path = os.path.join(self.config['data']['images_dir'], img_filename)
        
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
                'image_path': os.path.join(self.config['data']['images_dir'], img_info['file_name']),
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
        
        # Encode with CLIP Text Encoder (already batched internally)
        logger.info("Encoding grounded strings with CLIP...")
        embeddings = self.clip_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32  # Explicit batch size for efficiency
        )
        
        # Store in ChromaDB in batches to avoid memory issues
        logger.info("Storing in ChromaDB (batched)...")
        batch_size = 5000  # ChromaDB handles this well
        for i in range(0, len(image_ids), batch_size):
            end_idx = min(i + batch_size, len(image_ids))
            self.collections['grounded'].add(
                embeddings=embeddings[i:end_idx].tolist(),
                ids=image_ids[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            logger.info(f"  Inserted batch {i//batch_size + 1}: {end_idx}/{len(image_ids)}")
        
        logger.info(f"Indexed {len(image_ids)} grounded vectors")
    
    def index_vibe_layer(self, vibe_captions_path: str = "vibe_captions.json"):
        """
        Encode and store vibe captions in ChromaDB with batched inserts
        
        Args:
            vibe_captions_path: Path to vibe captions JSON file
        """
        logger.info("=== Indexing Vibe Layer (V_vibe) ===")
        
        # Load vibe captions
        vibe_data = load_json(vibe_captions_path)
        
        image_ids = []
        texts = []
        metadatas = []
        
        for img_id, caption in vibe_data.items():
            image_ids.append(str(img_id))
            texts.append(caption)
            
            # Get image path
            img_info = self.fp.loadImgs([int(img_id)])[0]
            img_path = os.path.join(self.config['data']['images_dir'], img_info['file_name'])
            
            metadatas.append({
                'text': caption,
                'image_path': img_path
            })
        
        # Encode with CLIP Text Encoder
        logger.info("Encoding vibe captions with CLIP...")
        embeddings = self.clip_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32  # Explicit batch size
        )
        
        # Store in ChromaDB in batches
        logger.info("Storing in ChromaDB (batched)...")
        batch_size = 5000
        for i in range(0, len(image_ids), batch_size):
            end_idx = min(i + batch_size, len(image_ids))
            self.collections['vibe'].add(
                embeddings=embeddings[i:end_idx].tolist(),
                ids=image_ids[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            logger.info(f"  Inserted batch {i//batch_size + 1}: {end_idx}/{len(image_ids)}")
        
        logger.info(f"Indexed {len(image_ids)} vibe vectors")
    
    def index_visual_layer(self, image_ids: List[int], batch_size: int = 32):
        """
        Encode and store raw images in ChromaDB with BATCHING for 3-5x speedup
        
        Args:
            image_ids: List of image IDs to process
            batch_size: Number of images to encode at once (default: 32)
        """
        logger.info("=== Indexing Visual Layer (V_img) ===")
        logger.info(f"Using batch size: {batch_size} for faster encoding")
        
        total_indexed = 0
        failed_ids = []
        
        # Process in batches
        for batch_start in tqdm(range(0, len(image_ids), batch_size), desc="Processing images"):
            batch_ids = image_ids[batch_start:batch_start + batch_size]
            
            batch_images = []
            batch_img_ids = []
            batch_metadatas = []
            
            for img_id in batch_ids:
                # Get image path
                img_info = self.fp.loadImgs([img_id])[0]
                img_filename = img_info['file_name']
                img_path = os.path.join(self.config['data']['images_dir'], img_filename)
                
                if not os.path.exists(img_path):
                    failed_ids.append(img_id)
                    continue
                
                try:
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
                # Batch encode with CLIP - MUCH faster!
                embeddings = self.clip_model.encode(batch_images, convert_to_numpy=True)
                
                # Store batch in ChromaDB
                self.collections['visual'].add(
                    embeddings=embeddings.tolist(),
                    ids=batch_img_ids,
                    metadatas=batch_metadatas
                )
                
                total_indexed += len(batch_img_ids)
                
            except Exception as e:
                logger.error(f"Batch encoding failed at {batch_start}: {e}")
                failed_ids.extend([int(i) for i in batch_img_ids])
        
        logger.info(f"Indexed {total_indexed} visual vectors")
        if failed_ids:
            logger.warning(f"Failed to process {len(failed_ids)} images")
    
    def build_index(self, num_images: int = None):
        """
        Main pipeline: Build all three vector indices
        
        Args:
            num_images: Number of images to process (None = use config)
        """
        if num_images is None:
            num_images = self.config['data']['num_images']
            if num_images == -1:
                num_images = len(self.fp.getImgIds())
        
        # Get image IDs
        logger.info(f"Selecting {num_images} images...")
        all_img_ids = self.fp.getImgIds()
        image_ids = all_img_ids[:num_images]
        
        logger.info(f"Processing {len(image_ids)} images")
        
        # Step 1: Generate grounded layer
        grounded_data = self.generate_grounded_vectors(image_ids)
        
        # Step 2: Index grounded layer
        self.index_grounded_layer(grounded_data)
        
        # Step 3: Index vibe layer (requires vibe_captions.json)
        vibe_path = "vibe_captions.json"
        if os.path.exists(vibe_path):
            self.index_vibe_layer(vibe_path)
        else:
            logger.warning(f"{vibe_path} not found. Run caption generation first.")
            logger.warning("Skipping vibe layer indexing...")
        
        # Step 4: Index visual layer
        self.index_visual_layer(image_ids)
        
        logger.info("=== Indexing Complete ===")
        logger.info("Collections created:")
        for key, collection in self.collections.items():
            count = collection.count()
            logger.info(f"  - {key}: {count} vectors")


def main():
    """Main execution"""
    indexer = MultiStreamIndexer()
    indexer.build_index()


if __name__ == "__main__":
    main()
