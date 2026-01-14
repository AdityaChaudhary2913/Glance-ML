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
        print(f"Loading CLIP model: {self.config['models']['clip_model']}")
        self.clip_model = SentenceTransformer(self.config['models']['clip_model'])
        
        # Initialize ChromaDB client
        persist_dir = self.config['chromadb']['persist_directory']
        ensure_dir(persist_dir)
        
        print(f"Initializing ChromaDB at {persist_dir}")
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
            print(f"Created collection: {name}")
        
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
        output_path: str = "grounded_data.json"
    ) -> Dict:
        """
        Generate V_fact: Grounded layer vectors from Fashionpedia + colors
        
        Args:
            image_ids: List of image IDs to process
            output_path: Path to save grounded strings
            
        Returns:
            Dictionary mapping image_id to grounded string
        """
        print("\n=== Generating Grounded Layer (V_fact) ===")
        grounded_data = {}
        
        for img_id in tqdm(image_ids, desc="Processing grounded strings"):
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
        
        # Save grounded data
        save_json(grounded_data, output_path)
        
        return grounded_data
    
    def index_grounded_layer(self, grounded_data: Dict):
        """
        Encode and store grounded strings in ChromaDB
        
        Args:
            grounded_data: Dictionary from generate_grounded_vectors
        """
        print("\n=== Indexing Grounded Layer ===")
        
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
        
        # Encode with CLIP Text Encoder
        print("Encoding grounded strings with CLIP...")
        embeddings = self.clip_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Store in ChromaDB
        print("Storing in ChromaDB...")
        self.collections['grounded'].add(
            embeddings=embeddings.tolist(),
            ids=image_ids,
            metadatas=metadatas
        )
        
        print(f"Indexed {len(image_ids)} grounded vectors")
    
    def index_vibe_layer(self, vibe_captions_path: str = "vibe_captions.json"):
        """
        Encode and store vibe captions in ChromaDB
        
        Args:
            vibe_captions_path: Path to vibe captions JSON file
        """
        print("\n=== Indexing Vibe Layer (V_vibe) ===")
        
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
        print("Encoding vibe captions with CLIP...")
        embeddings = self.clip_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Store in ChromaDB
        print("Storing in ChromaDB...")
        self.collections['vibe'].add(
            embeddings=embeddings.tolist(),
            ids=image_ids,
            metadatas=metadatas
        )
        
        print(f"Indexed {len(image_ids)} vibe vectors")
    
    def index_visual_layer(self, image_ids: List[int]):
        """
        Encode and store raw images in ChromaDB
        
        Args:
            image_ids: List of image IDs to process
        """
        print("\n=== Indexing Visual Layer (V_img) ===")
        
        ids_list = []
        embeddings_list = []
        metadatas = []
        
        for img_id in tqdm(image_ids, desc="Processing images"):
            # Get image path
            img_info = self.fp.loadImgs([img_id])[0]
            img_filename = img_info['file_name']
            img_path = os.path.join(self.config['data']['images_dir'], img_filename)
            
            if not os.path.exists(img_path):
                continue
            
            try:
                # Load and encode image with CLIP Image Encoder
                img = Image.open(img_path).convert('RGB')
                embedding = self.clip_model.encode(img, convert_to_numpy=True)
                
                ids_list.append(str(img_id))
                embeddings_list.append(embedding.tolist())
                metadatas.append({'image_path': img_path})
                
            except Exception as e:
                print(f"Error processing image {img_id}: {e}")
                continue
        
        # Store in ChromaDB
        print("Storing in ChromaDB...")
        self.collections['visual'].add(
            embeddings=embeddings_list,
            ids=ids_list,
            metadatas=metadatas
        )
        
        print(f"Indexed {len(ids_list)} visual vectors")
    
    def build_index(self, num_images: int = None):
        """
        Main pipeline: Build all three vector indices
        
        Args:
            num_images: Number of images to process (None = use config)
        """
        if num_images is None:
            num_images = self.config['data']['num_images']
        
        # Get image IDs
        print(f"\nSelecting {num_images} images...")
        all_img_ids = self.fp.getImgIds()
        image_ids = all_img_ids[:num_images]
        
        print(f"Processing {len(image_ids)} images")
        
        # Step 1: Generate grounded layer
        grounded_data = self.generate_grounded_vectors(image_ids)
        
        # Step 2: Index grounded layer
        self.index_grounded_layer(grounded_data)
        
        # Step 3: Index vibe layer (requires vibe_captions.json)
        vibe_path = "vibe_captions.json"
        if os.path.exists(vibe_path):
            self.index_vibe_layer(vibe_path)
        else:
            print(f"\nWarning: {vibe_path} not found. Run caption generation first.")
            print("Skipping vibe layer indexing...")
        
        # Step 4: Index visual layer
        self.index_visual_layer(image_ids)
        
        print("\n=== Indexing Complete ===")
        print(f"Collections created:")
        for key, collection in self.collections.items():
            count = collection.count()
            print(f"  - {key}: {count} vectors")


def main():
    """Main execution"""
    indexer = MultiStreamIndexer()
    indexer.build_index()


if __name__ == "__main__":
    main()
