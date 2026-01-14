"""
Utility functions for the Triple-Stream Fashion Search Engine
Includes JSON parsing, color extraction, and helper functions
"""

import json
import sys
import os
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import webcolors
from typing import List, Dict, Tuple, Optional
from pycocotools import mask as maskUtils
from tqdm import tqdm

# Add fashionpedia-api to path
sys.path.insert(0, '/workspace/fashionpedia-api-master')
from fashionpedia.fp import Fashionpedia


def closest_color_name(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB tuple to closest color name
    
    Args:
        rgb: RGB tuple (r, g, b) with values 0-255
        
    Returns:
        Color name string (e.g., "red", "blue")
    """
    min_colors = {}
    
    # Define common fashion colors
    fashion_colors = {
        'red': (255, 0, 0),
        'orange': (255, 165, 0),
        'yellow': (255, 255, 0),
        'green': (0, 128, 0),
        'blue': (0, 0, 255),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'brown': (165, 42, 42),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'gray': (128, 128, 128),
        'beige': (245, 245, 220),
        'navy': (0, 0, 128),
        'maroon': (128, 0, 0),
        'teal': (0, 128, 128),
        'olive': (128, 128, 0),
        'gold': (255, 215, 0),
        'silver': (192, 192, 192),
        'khaki': (240, 230, 140),
        'burgundy': (128, 0, 32)
    }
    
    for name, color_rgb in fashion_colors.items():
        r_c, g_c, b_c = color_rgb
        rd = (r_c - rgb[0]) ** 2
        gd = (g_c - rgb[1]) ** 2
        bd = (b_c - rgb[2]) ** 2
        min_colors[rd + gd + bd] = name
    
    return min_colors[min(min_colors.keys())]


def extract_dominant_colors(
    image_path: str,
    segmentation: List,
    is_crowd: int = 0,
    k: int = 3,
    min_pixels: int = 50
) -> List[str]:
    """
    Extract dominant colors from a segmented region using K-means clustering
    
    Args:
        image_path: Path to the image file
        segmentation: Segmentation mask (polygon or RLE format)
        is_crowd: Whether annotation is a crowd (0 or 1)
        k: Number of clusters for K-means
        min_pixels: Minimum pixels needed for color extraction
        
    Returns:
        List of color names (e.g., ["red", "blue"])
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return ["neutral"]
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Create binary mask from segmentation
        if isinstance(segmentation, list):
            if isinstance(segmentation[0], list):
                # Polygon format - maskUtils already imported at top
                rles = maskUtils.frPyObjects(segmentation, h, w)
                rle = maskUtils.merge(rles)
            else:
                # Already RLE format
                rle = segmentation
        else:
            rle = segmentation
            
        # Decode mask
        if isinstance(rle, dict):
            binary_mask = maskUtils.decode(rle)
        else:
            # Handle list format
            return ["neutral"]
        
        # Extract pixels in the masked region
        masked_pixels = img_rgb[binary_mask > 0]
        
        # Check if we have enough pixels
        if len(masked_pixels) < min_pixels:
            return ["neutral"]
    
        sample_ratio = 0.10 
        max_samples = 500     
        
        n_samples = min(int(len(masked_pixels) * sample_ratio), max_samples)
        n_samples = max(n_samples, min_pixels)  # At least min_pixels
        
        if len(masked_pixels) > n_samples:
            indices = np.random.choice(len(masked_pixels), n_samples, replace=False)
            sampled_pixels = masked_pixels[indices]
        else:
            sampled_pixels = masked_pixels
        
        # Apply K-means clustering on sampled pixels
        k_actual = min(k, len(sampled_pixels) // 20)
        if k_actual < 1:
            return ["neutral"]
            
        kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init=10)
        kmeans.fit(sampled_pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Count pixels in each cluster
        labels = kmeans.labels_
        label_counts = np.bincount(labels)
        
        # Sort colors by frequency
        sorted_indices = np.argsort(-label_counts)
        
        # Convert top colors to color names
        color_names = []
        for idx in sorted_indices[:2]:  # Top 2 colors
            rgb = tuple(colors[idx])
            color_name = closest_color_name(rgb)
            if color_name not in color_names:
                color_names.append(color_name)
        
        return color_names if color_names else ["neutral"]
        
    except Exception as e:
        print(f"Color extraction failed for {image_path}: {e}")
        return ["neutral"]


def load_fashionpedia_data(
    annotations_path: str,
    attributes_path: str
) -> Tuple[Fashionpedia, Dict]:
    """
    Load Fashionpedia dataset and attribute mappings
    
    Args:
        annotations_path: Path to instances_attributes JSON file
        attributes_path: Path to attributes JSON file
        
    Returns:
        Tuple of (Fashionpedia object, attributes dict)
    """
    print("Loading Fashionpedia dataset...")
    fp = Fashionpedia(annotations_path)
    
    print("Loading attribute mappings...")
    with open(attributes_path, 'r') as f:
        attr_data = json.load(f)
    
    return fp, attr_data


def build_grounded_string(
    fp: Fashionpedia,
    image_id: int,
    annotations: List[Dict],
    colors_list: List[List[str]],
    max_attributes: int = 3
) -> str:
    """
    Build structured description from Fashionpedia annotations + extracted colors
    
    Args:
        fp: Fashionpedia object
        image_id: Image ID
        annotations: List of annotation dictionaries for this image
        colors_list: List of color lists for each annotation
        max_attributes: Maximum number of attributes to include per garment
        
    Returns:
        Grounded description string
        Example: "A red jacket with hood and waterproof material. Blue jeans with distressed finish."
    """
    garment_descriptions = []
    
    for i, ann in enumerate(annotations):
        # Get category name
        category_id = ann['category_id']
        category_name = fp.cats[category_id]['name']
        
        # Get attribute names
        attribute_ids = ann.get('attribute_ids', [])
        attribute_names = []
        for attr_id in attribute_ids[:max_attributes]:  # Limit to top N attributes
            if attr_id in fp.attrs:
                attr_name = fp.attrs[attr_id]['name']
                attribute_names.append(attr_name)
        
        # Get extracted color
        colors = colors_list[i] if i < len(colors_list) else ["colored"]
        color_str = " and ".join(colors) if len(colors) > 1 else colors[0]
        
        # Build garment string
        # Pattern: "A {color} {category} with {attributes}"
        desc = f"A {color_str} {category_name}"
        
        if attribute_names:
            attr_str = ", ".join(attribute_names)
            desc += f" with {attr_str}"
        
        garment_descriptions.append(desc)
    
    # Join all garment descriptions
    if garment_descriptions:
        return ". ".join(garment_descriptions) + "."
    else:
        return "An outfit."


def get_image_annotations(
    fp: Fashionpedia,
    image_id: int
) -> List[Dict]:
    """
    Get all annotations for a given image
    
    Args:
        fp: Fashionpedia object
        image_id: Image ID
        
    Returns:
        List of annotation dictionaries
    """
    ann_ids = fp.getAnnIds(imgIds=[image_id])
    annotations = fp.loadAnns(ann_ids)
    return annotations


def save_json(data: Dict, filepath: str):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {filepath}")


def load_json(filepath: str) -> Dict:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_dir(directory: str):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)
