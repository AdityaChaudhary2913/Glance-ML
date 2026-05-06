"""
Generate vibe captions using BLIP-2 for scene and style understanding
"""

import sys
import os
sys.path.insert(0, '/workspace/fashionpedia-api-master')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import yaml
import torch
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from shared.utils import (
    load_fashionpedia_data, 
    save_json, 
    load_json,
    get_image_annotations,
    extract_dominant_colors,
    build_grounded_string
)
from shared.logger import caption_logger as logger


class VibeCaptionGenerator:
    """
    Generate scene/style captions using BLIP-2 with constrained prompting
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize caption generator
        
        Args:
            config_path: Path to config.yaml (None = auto-detect)
        """
        # Load configuration
        if config_path is None:
            # Get path relative to project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            config_path = os.path.join(project_root, 'shared', 'config.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get caption prompt from config
        self.caption_prompt = self.config['captioning']['prompt']
        
        # Initialize BLIP-2 for more stable captioning
        logger.info("Loading BLIP-2 model: Salesforce/blip2-opt-2.7b")
        
        # Set device and dtype first
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
        logger.info(f"Using device: {self.device}")
        
        # Get project root and make data paths absolute
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        annotations_path = os.path.join(project_root, self.config['data']['annotations_path'])
        attributes_path = os.path.join(project_root, self.config['data']['attributes_path'])
        
        # Load Fashionpedia data
        self.fp, self.attr_data = load_fashionpedia_data(annotations_path, attributes_path)
        
        # Store project root for later use
        self.project_root = project_root
    
    def generate_grounded_vectors(
        self,
        image_ids: List[int],
        output_path: str = None,
        checkpoint_interval: int = 500,
        resume: bool = True
    ) -> Dict:
        """
        Generate grounded layer vectors (V_fact) from Fashionpedia + colors
        This runs BEFORE caption generation to provide context
        
        Args:
            image_ids: List of image IDs to process
            output_path: Path to save grounded vectors (default: project_root/grounded_vectors.json)
            checkpoint_interval: Save progress every N images
            resume: Whether to resume from checkpoint
            
        Returns:
            Dictionary mapping image_id to grounded metadata
        """
        if output_path is None:
            output_path = os.path.join(self.project_root, "grounded_vectors.json")
        
        import time
        start_time = time.time()
        
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Generating Grounded Vectors (V_fact)")
        logger.info("="*60)
        logger.info(f"Processing {len(image_ids)} images")
        logger.info(f"Checkpoint interval: every {checkpoint_interval} images")
        
        grounded_data = {}
        checkpoint_path = output_path.replace('.json', '_checkpoint.json')
        
        # Resume from checkpoint if exists
        if resume and os.path.exists(checkpoint_path):
            try:
                grounded_data = load_json(checkpoint_path)
                logger.info(f"✓ Resumed from checkpoint: {len(grounded_data)} images done")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        # Filter already processed
        processed_ids = set(grounded_data.keys())
        remaining_ids = [img_id for img_id in image_ids if str(img_id) not in processed_ids]
        
        if len(remaining_ids) < len(image_ids):
            logger.info(f"Skipping {len(image_ids) - len(remaining_ids)} already processed")
        
        processed_count = 0
        
        # Process images sequentially (multiprocessing has pickling issues with nested functions)
        # Main speedup will come from Stage 2 (GPU batch processing)
        for img_id in tqdm(remaining_ids, desc="Generating grounded strings"):
            # Get annotations
            annotations = get_image_annotations(self.fp, img_id)
            if not annotations:
                continue
            
            # Extract colors for all garments
            img_info = self.fp.loadImgs([img_id])[0]
            img_path = os.path.join(self.project_root, self.config['data']['images_dir'], img_info['file_name'])
            
            colors_list = []
            for ann in annotations:
                colors = extract_dominant_colors(
                    image_path=img_path,
                    segmentation=ann.get('segmentation', []),
                    is_crowd=ann.get('iscrowd', 0),
                    k=self.config['color_extraction']['kmeans_clusters'],
                    min_pixels=self.config['color_extraction']['min_pixels']
                )
                colors_list.append(colors)
            
            # Build grounded string
            grounded_str = build_grounded_string(
                self.fp, img_id, annotations, colors_list, max_attributes=3
            )
            
            # Store metadata
            grounded_data[str(img_id)] = {
                'text': grounded_str,
                'image_path': img_path,
                'categories': [ann['category_id'] for ann in annotations],
                'colors': colors_list
            }
            
            processed_count += 1
            
            # Checkpoint
            if processed_count % checkpoint_interval == 0:
                save_json(grounded_data, checkpoint_path)
                logger.info(f"✓ Checkpoint: {len(grounded_data)} grounded vectors")
        
        # Save final
        save_json(grounded_data, output_path)
        
        total_time = time.time() - start_time
        throughput = len(grounded_data) / total_time if total_time > 0 else 0
        logger.info("\n✓ Grounded vector generation complete:")
        logger.info(f"  - Total: {len(grounded_data)} vectors")
        logger.info(f"  - Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"  - Throughput: {throughput:.2f} images/sec")
        logger.info(f"  - Output: {output_path}")
        
        # Remove checkpoint
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        return grounded_data
    
    def generate_captions_batch(self, image_paths: List[str], batch_size: int = 8) -> List[str]:
        """Generate vibe captions for a batch of images (legacy method, no context)
        
        Args:
            image_paths: List of image paths to process
            batch_size: Number of images to process at once (default: 8 for BLIP-2)
            
        Returns:
            List of captions corresponding to image_paths
        """
        captions = []
        
        for batch_start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_start:batch_start + batch_size]
            batch_images = []
            
            # Load all images in batch
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")
                    batch_images.append(None)
            
            # Filter out None values
            valid_images = [img for img in batch_images if img is not None]
            
            if not valid_images:
                captions.extend(["neutral setting, everyday wear"] * len(batch_images))
                continue
            
            try:
                # Process images sequentially with BLIP-2
                batch_captions = []
                
                for img in valid_images:
                    inputs = self.processor(
                        images=img,
                        text=self.caption_prompt,
                        return_tensors="pt"
                    ).to(self.device, self.torch_dtype)
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=100,
                            num_beams=3
                        )
                    
                    # Decode caption
                    caption = self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()
                    
                    # Remove prompt if present (BLIP-2 may echo the prompt)
                    if self.caption_prompt and caption.startswith(self.caption_prompt):
                        caption = caption[len(self.caption_prompt):].strip()
                    
                    if not caption:
                        caption = "neutral setting, everyday wear"
                    
                    batch_captions.append(caption)
                
                # Map back to original batch (accounting for failed loads)
                caption_idx = 0
                for img in batch_images:
                    if img is None:
                        captions.append("neutral setting, everyday wear")
                    else:
                        captions.append(batch_captions[caption_idx])
                        caption_idx += 1
                        
            except Exception as e:
                logger.warning(f"Batch caption generation failed: {e}")
                captions.extend(["neutral setting, everyday wear"] * len(batch_images))
        
        return captions
    
    def generate_captions_batch_with_context(
        self, 
        image_paths: List[str], 
        grounded_strings: List[str],
        batch_size: int = 8
    ) -> List[str]:
        """Batch generate captions WITH grounded context enrichment
        
        Uses BLIP-2 to generate scene captions and enriches them with grounded metadata
        
        Args:
            image_paths: List of image paths
            grounded_strings: List of grounded descriptions (from V_fact)
            batch_size: Batch size for processing (default: 8 for BLIP-2)
        
        Returns:
            List of captions enriched with Fashionpedia compositional grounding
        """
        captions = []
        
        for batch_start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_start:batch_start + batch_size]
            batch_grounded = grounded_strings[batch_start:batch_start + batch_size]
            batch_images = []
            valid_grounded = []
            
            # Load images - ensure no None values slip through
            for img_path, grounded_str in zip(batch_paths, batch_grounded):
                try:
                    img = Image.open(img_path).convert('RGB')
                    if img is not None:  # Double-check image loaded
                        batch_images.append(img)
                        valid_grounded.append(grounded_str if grounded_str else "")
                    else:
                        logger.warning(f"Image loaded as None: {img_path}")
                        batch_images.append(None)
                        valid_grounded.append(None)
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")
                    batch_images.append(None)
                    valid_grounded.append(None)
            
            # Filter valid entries - CRITICAL: keep ALL valid images
            # If an image loads but has no grounded string, use empty string (blind captioning fallback)
            valid_images = []
            valid_grounded_filtered = []
            for img, grounded in zip(batch_images, valid_grounded):
                if img is not None:  # Only check if image is valid
                    valid_images.append(img)
                    # Use empty string if grounded is None - enables blind captioning for this image
                    valid_grounded_filtered.append(grounded if grounded else "")
            
            if not valid_images:
                captions.extend(["neutral setting, everyday wear"] * len(batch_images))
                continue
            
            # Debug: Verify all valid_images are actually PIL Images
            for idx, img in enumerate(valid_images):
                if img is None or not hasattr(img, 'size'):
                    logger.error(f"Invalid image at index {idx}: {type(img)}")
                    raise ValueError(f"None image found in valid_images at index {idx}")
            
            try:
                # Debug: Log what we're about to process
                logger.debug(f"Processing batch: {len(valid_images)} images, {len(valid_grounded_filtered)} grounded strings")
                
                # PARALLEL BATCH PROCESSING FIX: Process all images in batch at once
                # Build contextualized prompts for each image
                batch_prompts = []
                for grounded_str in valid_grounded_filtered:
                    if grounded_str:
                        # Simplify grounded context - first 2 sentences only
                        grounded_parts = grounded_str.split('. ')[:2]
                        simplified_context = '. '.join(grounded_parts) + '.'
                        # Shorter, clearer prompt
                        prompt = f"Items: {simplified_context} Describe the setting and vibe:"
                    else:
                        prompt = "Describe the setting and vibe:"
                    batch_prompts.append(prompt)
                
                # Process entire batch in parallel on GPU
                inputs = self.processor(
                    images=valid_images,
                    text=batch_prompts,
                    return_tensors="pt",
                    padding=True
                ).to(self.device, self.torch_dtype)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=50,  # Shorter to avoid rambling
                        num_beams=3,
                        repetition_penalty=1.5,  # Penalize repetition
                        no_repeat_ngram_size=3  # Prevent 3-word phrases from repeating
                    )
                
                # Decode all captions in batch with robust cleaning
                batch_captions = []
                for idx, gen_ids in enumerate(generated_ids):
                    caption = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
                    
                    # Remove prompt if present
                    prompt = batch_prompts[idx]
                    if prompt and caption.startswith(prompt):
                        caption = caption[len(prompt):].strip()
                    
                    # Robust cleaning - remove prompt patterns
                    patterns_to_remove = [
                        "Items:", "Describe the setting and vibe:", "Context:",
                        "Question:", "Answer:", "Format:", "Examples:"
                    ]
                    for pattern in patterns_to_remove:
                        if pattern in caption:
                            parts = caption.split(pattern, 1)
                            if len(parts) > 1:
                                caption = parts[1].strip()
                    
                    # ENHANCED repetition detection
                    parts = caption.replace(',', '.').split('.')
                    parts = [p.strip() for p in parts if p.strip()]
                    
                    if len(parts) > 2:
                        seen = {}
                        unique_parts = []
                        for part in parts:
                            normalized = ' '.join(part.split())
                            if normalized and normalized not in seen:
                                seen[normalized] = True
                                unique_parts.append(part)
                        
                        # If we removed a lot of repetition, reconstruct
                        if len(unique_parts) < len(parts) * 0.7:
                            caption = '. '.join(unique_parts[:3])
                            if caption and not caption.endswith('.'):
                                caption += '.'
                    
                    # Validate and filter nonsense
                    if not caption or len(caption) < 10:
                        caption = "neutral setting, everyday wear"
                    
                    nonsense_indicators = [
                        "I don't know if this is the right place",
                        "does anyone know if there's a way",
                        "can't seem to get it to work"
                    ]
                    for indicator in nonsense_indicators:
                        if indicator.lower() in caption.lower():
                            caption = "neutral setting, everyday wear"
                            break
                    
                    # Format with grounded context
                    grounded_str = valid_grounded_filtered[idx]
                    if grounded_str:
                        enriched_caption = f"{grounded_str} | Scene: {caption}"
                        batch_captions.append(enriched_caption)
                    else:
                        batch_captions.append(caption)
                
                # Map back to original batch
                caption_idx = 0
                for img in batch_images:
                    if img is None:
                        captions.append("neutral setting, everyday wear")
                    else:
                        captions.append(batch_captions[caption_idx])
                        caption_idx += 1
                        
            except Exception as e:
                logger.error(f"Batch caption generation with context failed: {e}")
                logger.error(f"  Batch details: {len(batch_images)} total, {len(valid_images)} valid")
                import traceback
                logger.error(f"  Traceback: {traceback.format_exc()}")
                captions.extend(["neutral setting, everyday wear"] * len(batch_images))
        
        return captions
    
    def generate_captions(
        self,
        num_images: int = None,
        output_path: str = None,
        checkpoint_interval: int = 500,
        resume: bool = True,
        batch_size: int = 8,
        auto_generate_grounded: bool = True
    ) -> dict:
        """
        Generate contextual vibe captions - SELF-CONTAINED pipeline
        Automatically generates grounded vectors first, then creates captions
        
        Args:
            num_images: Number of images to process (None = use config)
            output_path: Path to save captions JSON (default: project_root/vibe_captions.json)
            checkpoint_interval: Save progress every N images
            resume: Whether to resume from existing checkpoint
            batch_size: Number of images to process at once (default: 8 for BLIP-2)
            auto_generate_grounded: Whether to auto-generate grounded vectors (default: True)
            
        Returns:
            Dictionary mapping image_id to caption
        """
        if num_images is None:
            num_images = self.config['data']['num_images']
            if num_images == -1:
                num_images = len(self.fp.getImgIds())
        
        if output_path is None:
            output_path = os.path.join(self.project_root, "vibe_captions.json")
        
        import time
        pipeline_start = time.time()
        
        logger.info("\n" + "="*70)
        logger.info("CAPTION GENERATOR: Context-Aware Vibe Caption Pipeline")
        logger.info("="*70)
        logger.info(f"Target: {num_images} images")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Checkpoint interval: every {checkpoint_interval} images")
        logger.info("")
        
        # Get image IDs
        all_img_ids = self.fp.getImgIds()
        image_ids = all_img_ids[:num_images]
        
        # Step 1: Generate or load grounded vectors
        grounded_path = os.path.join(self.project_root, "grounded_vectors.json")
        grounded_data = {}
        stage1_start = time.time()
        
        if auto_generate_grounded:
            if os.path.exists(grounded_path) and resume:
                logger.info(f"Found existing grounded vectors at: {grounded_path}")
                logger.info("Loading grounded vectors...")
                grounded_data = load_json(grounded_path)
                logger.info(f"✓ Loaded {len(grounded_data)} grounded vectors")
            else:
                grounded_data = self.generate_grounded_vectors(
                    image_ids,
                    output_path=grounded_path,
                    checkpoint_interval=checkpoint_interval,
                    resume=resume
                )
        else:
            if os.path.exists(grounded_path):
                grounded_data = load_json(grounded_path)
                logger.info(f"✓ Loaded {len(grounded_data)} grounded vectors")
            else:
                logger.warning("⚠️  No grounded vectors found - falling back to blind captioning")
        
        stage1_time = time.time() - stage1_start
        
        # Step 2: Generate context-aware captions
        stage2_start = time.time()
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Generating Context-Aware Captions (V_vibe)")
        logger.info("="*60)
        
        # Get image IDs
        all_img_ids = self.fp.getImgIds()
        image_ids = all_img_ids[:num_images]
        
        # Resume from checkpoint if exists
        captions = {}
        checkpoint_path = output_path.replace('.json', '_checkpoint.json')
        
        if resume and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    captions = json.load(f)
                logger.info(f"✓ Resumed from checkpoint: {len(captions)} captions already done")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        # Filter out already processed images
        processed_ids = set(captions.keys())
        remaining_ids = [img_id for img_id in image_ids if str(img_id) not in processed_ids]
        
        if len(remaining_ids) < len(image_ids):
            logger.info(f"Skipping {len(image_ids) - len(remaining_ids)} already processed images")
        
        # Track batch timing
        batch_times = []
        total_batches = (len(remaining_ids) + batch_size - 1) // batch_size
        
        # Process in batches for speedup
        for batch_num, batch_start in enumerate(tqdm(range(0, len(remaining_ids), batch_size), desc="Generating captions"), 1):
            batch_start_time = time.time()
            batch_ids = remaining_ids[batch_start:batch_start + batch_size]
            
            # Collect image paths and grounded strings for this batch
            batch_img_paths = []
            batch_img_ids = []
            batch_grounded_strings = []
            
            for img_id in batch_ids:
                img_info = self.fp.loadImgs([img_id])[0]
                img_filename = img_info['file_name']
                img_path = os.path.join(self.project_root, self.config['data']['images_dir'], img_filename)
                
                if os.path.exists(img_path):
                    batch_img_paths.append(img_path)
                    batch_img_ids.append(img_id)
                    
                    # Get grounded string if available
                    grounded_str = grounded_data.get(str(img_id), {}).get('text', '')
                    batch_grounded_strings.append(grounded_str)
                else:
                    logger.warning(f"Image not found: {img_path}")
            
            if not batch_img_paths:
                continue
            
            # Use context-aware batch generation if we have grounded data
            if grounded_data and any(batch_grounded_strings):
                batch_captions = self.generate_captions_batch_with_context(
                    batch_img_paths, 
                    batch_grounded_strings,
                    batch_size
                )
            else:
                # Fallback to blind captioning
                batch_captions = self.generate_captions_batch(batch_img_paths, batch_size)
            
            # Store captions
            for img_id, caption in zip(batch_img_ids, batch_captions):
                captions[str(img_id)] = caption
            
            # Track batch timing
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Calculate ETA
            if len(batch_times) >= 3:  # Need at least 3 batches for stable estimate
                avg_batch_time = sum(batch_times[-10:]) / len(batch_times[-10:])  # Use last 10 batches
                remaining_batches = total_batches - batch_num
                eta_seconds = avg_batch_time * remaining_batches
                
                if batch_num % 10 == 0:  # Log every 10 batches
                    logger.info(f"  Batch {batch_num}/{total_batches} - {len(batch_captions)} captions in {batch_time:.1f}s - ETA: {eta_seconds/60:.1f} min")
            
            # Save checkpoint periodically
            if len(captions) % checkpoint_interval < batch_size:
                save_json(captions, checkpoint_path)
                logger.info(f"✓ Checkpoint saved: {len(captions)} captions")
        
        # Diversity check
        unique_settings = len(set([c.split(',')[0].strip() for c in captions.values()]))
        logger.info(f"Diversity check: {unique_settings} unique settings out of {len(captions)} captions")
        
        min_unique = self.config['captioning']['min_unique_settings']
        if unique_settings < min_unique:
            logger.warning(f"Low diversity (< {min_unique} unique settings)")
            logger.warning("Consider adjusting the prompt or increasing image diversity")
        else:
            logger.info(f"✓ Good diversity (>= {min_unique} unique settings)")
        
        # Save final captions
        save_json(captions, output_path)
        
        # Calculate stage 2 timing
        stage2_time = time.time() - stage2_start
        total_pipeline_time = time.time() - pipeline_start
        
        # Remove checkpoint file after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info("✓ Removed checkpoint file (full run completed)")
        
        # Performance summary
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*60)
        logger.info(f"Stage 1 (Grounded Vectors): {stage1_time:.1f}s ({stage1_time/60:.1f} min)")
        logger.info(f"Stage 2 (Caption Generation): {stage2_time:.1f}s ({stage2_time/60:.1f} min)")
        logger.info(f"Total Pipeline Time: {total_pipeline_time:.1f}s ({total_pipeline_time/60:.1f} min)")
        
        if len(captions) > 0 and stage2_time > 0:
            caption_throughput = len(captions) / stage2_time
            logger.info(f"Caption Throughput: {caption_throughput:.2f} captions/sec")
            
            if batch_times:
                avg_batch_time = sum(batch_times) / len(batch_times)
                logger.info(f"Average Batch Time: {avg_batch_time:.2f}s ({batch_size/avg_batch_time:.1f} images/sec)")
        
        # Print sample captions
        logger.info("\nSample captions:")
        for i, (img_id, caption) in enumerate(list(captions.items())[:5]):
            logger.info(f"  {img_id}: {caption}")
        
        return captions


def main():
    """Main execution - runs complete caption generation pipeline"""
    import time
    main_start = time.time()
    
    logger.info("="*70)
    logger.info("Starting Self-Contained Caption Generation Pipeline")
    logger.info("="*70)
    
    generator = VibeCaptionGenerator()
    
    # This will:
    # 1. Generate grounded vectors (V_fact)
    # 2. Generate context-aware captions (V_vibe) using grounded vectors
    captions = generator.generate_captions(
        auto_generate_grounded=True,
        resume=True  # Resume from checkpoints if interrupted
    )
    
    total_time = time.time() - main_start
    
    logger.info("")
    logger.info("="*70)
    logger.info("✓ Caption Generation Pipeline Complete")
    logger.info("="*70)
    logger.info(f"Generated {len(captions)} contextual vibe captions")
    logger.info(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} min, {total_time/3600:.2f} hours)")
    logger.info("Output files:")
    logger.info("  - grounded_vectors.json (V_fact metadata)")
    logger.info("  - vibe_captions.json (context-aware captions)")
    logger.info("")
    logger.info("Next step: Run indexer to build vector database")
    logger.info("  cd indexer && python indexer.py")


if __name__ == "__main__":
    main()
