"""
Retriever: Intent-Aware Dynamic Search Engine
Implements query-time weighted fusion of three vector streams
"""

import sys
sys.path.insert(0, '/workspace/fashionpedia-api-master')

import os
import json
import yaml
import numpy as np
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
import chromadb
        
from logger import retriever_logger as logger


class TripleStreamRetriever:
    """
    Dynamic search with query-time weighting across three vector streams
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize retriever
        
        Args:
            config_path: Path to config.yaml
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize CLIP model
        logger.info(f"Loading CLIP model: {self.config['models']['clip_model']}")
        self.clip_model = SentenceTransformer(self.config['models']['clip_model'])
        
        # Initialize ChromaDB client
        persist_dir = self.config['chromadb']['persist_directory']
        logger.info(f"Loading ChromaDB from {persist_dir}")
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Load collections
        collection_names = self.config['chromadb']['collections']
        self.collections = {}
        
        for key, name in collection_names.items():
            try:
                self.collections[key] = self.client.get_collection(name)
                count = self.collections[key].count()
                logger.info(f"Loaded collection '{name}': {count} vectors")
            except Exception as e:
                logger.warning(f"Could not load collection '{name}': {e}")
        
        # Load expansion rules
        self.expansion_rules = self.config['expansion_rules']
        
        # Load weight presets
        self.weight_presets = self.config['weight_presets']
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with synonym keywords
        
        Args:
            query: Original query string
            
        Returns:
            Expanded query string
        """
        expanded = query
        query_lower = query.lower()
        
        for keyword, expansion in self.expansion_rules.items():
            if keyword in query_lower:
                expanded += f" {expansion}"
        
        return expanded
    
    def normalize_scores(self, distances: List[float]) -> np.ndarray:
        """
        Convert ChromaDB L2 distances to similarities in [0, 1] range
        Lower distance = higher similarity
        
        Args:
            distances: List of L2 distances from ChromaDB
            
        Returns:
            Normalized similarity scores
        """
        distances = np.array(distances)
        
        # Handle edge case: all distances are 0
        if distances.max() == 0:
            return np.ones_like(distances)
        
        # Invert and normalize: lower distance = higher similarity
        similarities = 1 - (distances / distances.max())
        
        # Ensure in [0, 1] range
        similarities = np.clip(similarities, 0, 1)
        
        return similarities
    
    def dynamic_search(
        self,
        query: str,
        alpha: float = 0.33,
        beta: float = 0.33,
        gamma: float = 0.33,
        top_k: int = None,
        expand: bool = True,
        preset: Optional[str] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Execute dynamic search with query-time weighting
        
        Args:
            query: Search query string
            alpha: Weight for grounded layer (0-1)
            beta: Weight for vibe layer (0-1)
            gamma: Weight for visual layer (0-1)
            top_k: Number of results to return (None = use config)
            expand: Whether to apply query expansion
            preset: Use a weight preset from config (overrides alpha/beta/gamma)
            
        Returns:
            List of (image_id, final_score, individual_scores_dict) tuples
        """
        if top_k is None:
            top_k = self.config['retrieval']['top_k']
        
        # Use preset if specified
        if preset and preset in self.weight_presets:
            weights = self.weight_presets[preset]
            alpha = weights['alpha']
            beta = weights['beta']
            gamma = weights['gamma']
            logger.info(f"Using preset '{preset}': α={alpha}, β={beta}, γ={gamma}")
        
        # Normalize weights to sum to 1.0
        total = alpha + beta + gamma
        if total > 0:
            alpha, beta, gamma = alpha/total, beta/total, gamma/total
        
        # Expand query if enabled
        original_query = query
        if expand:
            query = self.expand_query(query)
            if query != original_query:
                logger.info(f"Expanded query: '{original_query}' → '{query}'")
        
        # Encode query with CLIP Text Encoder
        query_embedding = self.clip_model.encode(query, convert_to_numpy=True)
        
        # Query all collections
        n_results = self.config['retrieval']['query_n_results']
        
        results = {}
        
        # Query grounded collection
        if 'grounded' in self.collections and alpha > 0:
            grounded_results = self.collections['grounded'].query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            grounded_ids = grounded_results['ids'][0]
            grounded_distances = grounded_results['distances'][0]
            grounded_scores = self.normalize_scores(grounded_distances)
            
            for img_id, score in zip(grounded_ids, grounded_scores):
                if img_id not in results:
                    results[img_id] = {'grounded': 0.0, 'vibe': 0.0, 'visual': 0.0}
                results[img_id]['grounded'] = float(score)
        
        # Query vibe collection
        if 'vibe' in self.collections and beta > 0:
            vibe_results = self.collections['vibe'].query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            vibe_ids = vibe_results['ids'][0]
            vibe_distances = vibe_results['distances'][0]
            vibe_scores = self.normalize_scores(vibe_distances)
            
            for img_id, score in zip(vibe_ids, vibe_scores):
                if img_id not in results:
                    results[img_id] = {'grounded': 0.0, 'vibe': 0.0, 'visual': 0.0}
                results[img_id]['vibe'] = float(score)
        
        # Query visual collection
        if 'visual' in self.collections and gamma > 0:
            visual_results = self.collections['visual'].query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            visual_ids = visual_results['ids'][0]
            visual_distances = visual_results['distances'][0]
            visual_scores = self.normalize_scores(visual_distances)
            
            for img_id, score in zip(visual_ids, visual_scores):
                if img_id not in results:
                    results[img_id] = {'grounded': 0.0, 'vibe': 0.0, 'visual': 0.0}
                results[img_id]['visual'] = float(score)
        
        # Compute final weighted scores
        final_results = []
        for img_id, scores in results.items():
            final_score = (
                alpha * scores['grounded'] +
                beta * scores['vibe'] +
                gamma * scores['visual']
            )
            final_results.append((img_id, final_score, scores))
        
        # Sort by final score (descending)
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return final_results[:top_k]
    
    def vanilla_clip_search(self, query: str, top_k: int = None) -> List[str]:
        """
        Vanilla CLIP baseline (only visual layer)
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of image IDs
        """
        if top_k is None:
            top_k = self.config['retrieval']['top_k']
        
        if 'visual' not in self.collections:
            raise ValueError("Visual collection not loaded")
        
        # Encode query
        query_embedding = self.clip_model.encode(query, convert_to_numpy=True)
        
        # Query visual collection only
        visual_results = self.collections['visual'].query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        return visual_results['ids'][0]
    
    def get_image_metadata(self, image_id: str) -> Dict:
        """
        Retrieve metadata for an image from all collections
        
        Args:
            image_id: Image ID
            
        Returns:
            Dictionary with metadata from all collections
        """
        metadata = {}
        
        # Get from grounded collection
        if 'grounded' in self.collections:
            try:
                result = self.collections['grounded'].get(ids=[image_id])
                if result['metadatas']:
                    metadata['grounded_text'] = result['metadatas'][0].get('text', '')
                    metadata['image_path'] = result['metadatas'][0].get('image_path', '')
            except:
                pass
        
        # Get from vibe collection
        if 'vibe' in self.collections:
            try:
                result = self.collections['vibe'].get(ids=[image_id])
                if result['metadatas']:
                    metadata['vibe_text'] = result['metadatas'][0].get('text', '')
            except:
                pass
        
        # Get from visual collection
        if 'visual' in self.collections:
            try:
                result = self.collections['visual'].get(ids=[image_id])
                if result['metadatas']:
                    if 'image_path' not in metadata:
                        metadata['image_path'] = result['metadatas'][0].get('image_path', '')
            except:
                pass
        
        return metadata
    
    def print_results(self, query: str, results: List[Tuple[str, float, Dict]], max_text_len: int = 80):
        """
        Pretty print search results
        
        Args:
            query: Original query
            results: Results from dynamic_search
            max_text_len: Maximum text length to display
        """
        logger.info(f"\n{'='*100}")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*100}\n")
        
        for i, (img_id, score, individual_scores) in enumerate(results, 1):
            metadata = self.get_image_metadata(img_id)
            
            logger.info(f"Rank {i}: Image ID {img_id}")
            logger.info(f"  Final Score: {score:.4f}")
            logger.info(f"  Stream Scores: G={individual_scores['grounded']:.3f}, "
                  f"V={individual_scores['vibe']:.3f}, I={individual_scores['visual']:.3f}")
            
            if 'image_path' in metadata:
                logger.info(f"  Path: {metadata['image_path']}")
            
            if 'grounded_text' in metadata:
                text = metadata['grounded_text']
                display_text = text[:max_text_len] + "..." if len(text) > max_text_len else text
                logger.info(f"  Grounded: {display_text}")
            
            if 'vibe_text' in metadata:
                text = metadata['vibe_text']
                display_text = text[:max_text_len] + "..." if len(text) > max_text_len else text
                logger.info(f"  Vibe: {display_text}")
            
            logger.info("")


def main():
    """Demo usage"""
    retriever = TripleStreamRetriever()
    
    # Test queries from assignment
    test_queries = [
        ("A person in a bright yellow raincoat", "attribute_specific"),
        ("Professional business attire inside a modern office", "contextual_place"),
        ("Someone wearing a blue shirt sitting on a park bench", "complex_semantic"),
        ("Casual weekend outfit for a city walk", "style_inference"),
        ("A red tie and a white shirt in a formal setting", "compositional")
    ]
    
    for query, preset in test_queries:
        results = retriever.dynamic_search(query, preset=preset)
        retriever.print_results(query, results)


if __name__ == "__main__":
    main()
