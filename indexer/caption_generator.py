"""
Generate vibe captions using BLIP-2 for scene and style understanding
"""

import sys
import os
sys.path.insert(0, '/workspace/fashionpedia-api-master')
# Add parent directory to path for shared modules
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
        
        # Initialize BLIP-2
        logger.info(f"Loading BLIP-2 model: {self.config['models']['blip2_model']}")
        self.processor = Blip2Processor.from_pretrained(self.config['models']['blip2_model'])
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.config['models']['blip2_model'],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Get project root and make data paths absolute
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        annotations_path = os.path.join(project_root, self.config['data']['annotations_path'])
        attributes_path = os.path.join(project_root, self.config['data']['attributes_path'])
        
        # Load Fashionpedia data
        self.fp, self.attr_data = load_fashionpedia_data(annotations_path, attributes_path)
        
        # Store project root for later use
        self.project_root = project_root
        
        # Get caption prompt
        self.prompt = self.config['captioning']['prompt']
        logger.info(f"Caption prompt: {self.prompt}")
    
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
        logger.info(f"✓ Saved {len(grounded_data)} grounded vectors to: {output_path}")
        
        # Remove checkpoint
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        return grounded_data
    
    def generate_caption(self, image_path: str) -> str:
        """Generate a vibe caption for a single image (legacy method, no context)"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Prepare inputs WITHOUT the prompt text (BLIP-2 issue)
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
            
            # Generate caption with the prompt as text input
            prompt_text = "Question: What is the setting and style? Answer:"
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    num_beams=3,
                    min_length=5,
                    temperature=0.7,
                    do_sample=False
                )
            
            # Decode caption
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up caption - remove prompt if echoed
            caption = caption.replace(prompt_text, "").strip()
            if not caption or caption == self.prompt:
                return "neutral setting, everyday wear"
            
            return caption
            
        except Exception as e:
            logger.warning(f"Caption generation failed for {image_path}: {e}")
            return "neutral setting, everyday wear"
    
    def generate_caption_with_context(self, image_path: str, grounded_str: str) -> str:
        """Generate a vibe caption WITH knowledge of what's in the image
        
        Args:
            image_path: Path to the image
            grounded_str: Pre-computed grounded description (from V_fact)
                         Example: "A red blazer with wool. A blue jeans with slim fit."
        
        Returns:
            Scene and vibe caption grounded in actual garments
        """
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Build context-aware prompt using grounded string
            prompt_text = f"""The image contains: {grounded_str}

Question: Given these specific items, what is the scene setting (indoor/outdoor, location) and overall style vibe (formal/casual/athletic)? Answer in format: [Setting], [Vibe].
Answer:"""
            
            # Prepare inputs with the prompt
            inputs = self.processor(
                images=image,
                text=prompt_text,
                return_tensors="pt"
            ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=40,
                    num_beams=3,
                    min_length=10,
                    temperature=0.7,
                    do_sample=False
                )
            
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up - remove the long prompt echo
            caption = caption.split("Answer:")[-1].strip()
            
            if not caption:
                return "neutral setting, everyday wear"
            
            return caption
            
        except Exception as e:
            logger.warning(f"Context-aware caption generation failed for {image_path}: {e}")
            return "neutral setting, everyday wear"
    
    def generate_captions_batch(self, image_paths: List[str], batch_size: int = 16) -> List[str]:
        """Generate vibe captions for a batch of images (legacy method, no context)
        
        Args:
            image_paths: List of image paths to process
            batch_size: Number of images to process at once (default: 16)
            
        Returns:
            List of captions corresponding to image_paths
        """
        captions = []
        prompt_text = "Question: What is the setting and style? Answer:"
        
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
                # Batch process with BLIP-2
                inputs = self.processor(
                    images=valid_images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=30,
                        num_beams=3,
                        min_length=5,
                        temperature=0.7,
                        do_sample=False
                    )
                
                # Decode all captions
                batch_captions = []
                for gen_id in generated_ids:
                    caption = self.processor.decode(gen_id, skip_special_tokens=True)
                    caption = caption.replace(prompt_text, "").strip()
                    
                    if not caption or caption == self.prompt:
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
        batch_size: int = 16
    ) -> List[str]:
        """Batch generate captions WITH grounded context (5-10x faster)
        
        Args:
            image_paths: List of image paths
            grounded_strings: List of grounded descriptions (from V_fact)
            batch_size: Batch size for processing
        
        Returns:
            List of contextual vibe captions
        """
        captions = []
        
        for batch_start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_start:batch_start + batch_size]
            batch_grounded = grounded_strings[batch_start:batch_start + batch_size]
            batch_images = []
            batch_prompts = []
            
            # Load images and build prompts
            for img_path, grounded_str in zip(batch_paths, batch_grounded):
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(img)
                    
                    prompt = f"""The image contains: {grounded_str}

Question: Given these items, what is the scene setting and style vibe? Format: [Setting], [Vibe].
Answer:"""
                    batch_prompts.append(prompt)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")
                    batch_images.append(None)
                    batch_prompts.append(None)
            
            # Filter valid entries
            valid_data = [(img, prompt) for img, prompt in zip(batch_images, batch_prompts) 
                          if img is not None]
            
            if not valid_data:
                captions.extend(["neutral setting, everyday wear"] * len(batch_images))
                continue
            
            valid_images = [x[0] for x in valid_data]
            valid_prompts = [x[1] for x in valid_data]
            
            try:
                # Batch process with context
                inputs = self.processor(
                    images=valid_images,
                    text=valid_prompts,
                    return_tensors="pt",
                    padding=True
                ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=40,
                        num_beams=3,
                        min_length=10,
                        temperature=0.7,
                        do_sample=False
                    )
                
                # Decode captions
                batch_captions = []
                for gen_id in generated_ids:
                    caption = self.processor.decode(gen_id, skip_special_tokens=True)
                    caption = caption.split("Answer:")[-1].strip()
                    
                    if not caption:
                        caption = "neutral setting, everyday wear"
                    
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
                logger.warning(f"Batch caption generation with context failed: {e}")
                captions.extend(["neutral setting, everyday wear"] * len(batch_images))
        
        return captions
    
    def generate_captions(
        self,
        num_images: int = None,
        output_path: str = None,
        checkpoint_interval: int = 500,
        resume: bool = True,
        batch_size: int = 16,
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
            batch_size: Number of images to process at once (default: 16)
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
        
        # Step 2: Generate context-aware captions
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
        
        # Process in batches for speedup
        for batch_start in tqdm(range(0, len(remaining_ids), batch_size), desc="Generating captions"):
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
        
        # Remove checkpoint file after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info("✓ Removed checkpoint file (full run completed)")
        
        # Print sample captions
        logger.info("Sample captions:")
        for i, (img_id, caption) in enumerate(list(captions.items())[:5]):
            logger.info(f"  {img_id}: {caption}")
        
        return captions


def main():
    """Main execution - runs complete caption generation pipeline"""
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
    
    logger.info("")
    logger.info("="*70)
    logger.info("✓ Caption Generation Pipeline Complete")
    logger.info("="*70)
    logger.info(f"Generated {len(captions)} contextual vibe captions")
    logger.info("Output files:")
    logger.info("  - grounded_vectors.json (V_fact metadata)")
    logger.info("  - vibe_captions.json (context-aware captions)")
    logger.info("")
    logger.info("Next step: Run indexer to build vector database")
    logger.info("  cd indexer && python indexer.py")


if __name__ == "__main__":
    main()
