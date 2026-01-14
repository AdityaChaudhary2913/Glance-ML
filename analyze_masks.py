"""
Analyze actual mask sizes in Fashionpedia dataset to calibrate sampling heuristics
"""
import sys
sys.path.insert(0, '/workspace/fashionpedia-api-master')

import json
import numpy as np
from pycocotools import mask as maskUtils
from utils import load_fashionpedia_data

# Load dataset
fp, attr_data = load_fashionpedia_data(
    "data/instances_attributes_train2020.json",
    "data/attributes_train2020.json"
)

# Get image IDs
img_ids = fp.getImgIds()

# Sample 100 random images
sample_size = 100
sampled_ids = np.random.choice(img_ids[:2500], min(sample_size, len(img_ids[:2500])), replace=False)

mask_sizes = []

print(f"Analyzing {len(sampled_ids)} sample images...")

for img_id in sampled_ids:
    ann_ids = fp.getAnnIds(imgIds=[img_id])
    anns = fp.loadAnns(ann_ids)
    
    img_info = fp.loadImgs([img_id])[0]
    h, w = img_info['height'], img_info['width']
    
    for ann in anns:
        segmentation = ann['segmentation']
        
        try:
            # Decode mask
            if isinstance(segmentation, list):
                if isinstance(segmentation[0], list):
                    # Polygon format
                    rles = maskUtils.frPyObjects(segmentation, h, w)
                    rle = maskUtils.merge(rles)
                else:
                    continue
            else:
                rle = segmentation
                
            if isinstance(rle, dict):
                binary_mask = maskUtils.decode(rle)
                pixel_count = np.sum(binary_mask > 0)
                mask_sizes.append(pixel_count)
        except:
            continue

# Statistics
mask_sizes = np.array(mask_sizes)
print(f"\n=== Mask Size Statistics ===")
print(f"Total masks analyzed: {len(mask_sizes)}")
print(f"Min pixels: {mask_sizes.min()}")
print(f"Max pixels: {mask_sizes.max()}")
print(f"Mean: {mask_sizes.mean():.0f}")
print(f"Median: {np.median(mask_sizes):.0f}")
print(f"25th percentile: {np.percentile(mask_sizes, 25):.0f}")
print(f"75th percentile: {np.percentile(mask_sizes, 75):.0f}")
print(f"90th percentile: {np.percentile(mask_sizes, 90):.0f}")
print(f"95th percentile: {np.percentile(mask_sizes, 95):.0f}")

# Test different sampling strategies
print(f"\n=== Sampling Strategy Analysis ===")
for ratio in [0.05, 0.10, 0.15, 0.20]:
    for cap in [200, 300, 500, 1000]:
        samples = np.minimum(mask_sizes * ratio, cap)
        samples = np.maximum(samples, 50)  # min_pixels
        avg_samples = samples.mean()
        print(f"Ratio={ratio:.0%}, Cap={cap:4d} → Avg samples/mask: {avg_samples:.0f}")
