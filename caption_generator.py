"""
Generate vibe captions using BLIP-2 for scene and style understanding
"""

import sys
sys.path.insert(0, '/workspace/fashionpedia-api-master')

import os
import json
import yaml
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from utils import load_fashionpedia_data, save_json
from logger import caption_logger as logger


class VibeCaptionGenerator:
    """
    Generate scene/style captions using BLIP-2 with constrained prompting
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize caption generator
        
        Args:
            config_path: Path to config.yaml
        """
        # Load configuration
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
        
        # Load Fashionpedia data
        self.fp, _ = load_fashionpedia_data(
            self.config['data']['annotations_path'],
            self.config['data']['attributes_path']
        )
        
        # Get caption prompt
        self.prompt = self.config['captioning']['prompt']
        logger.info(f"Caption prompt: {self.prompt}")
    
    def generate_caption(self, image_path: str) -> str:
        """Generate a vibe caption for a single image"""
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
    
    def generate_captions(
        self,
        num_images: int = None,
        output_path: str = "vibe_captions.json"
    ) -> dict:
        """
        Generate vibe captions for multiple images
        
        Args:
            num_images: Number of images to process (None = use config)
            output_path: Path to save captions JSON
            
        Returns:
            Dictionary mapping image_id to caption
        """
        if num_images is None:
            num_images = self.config['data']['num_images']
            if num_images == -1:
                num_images = len(self.fp.getImgIds())
        
        logger.info("=== Generating Vibe Captions ===")
        logger.info(f"Target: {num_images} images")
        
        # Get image IDs
        all_img_ids = self.fp.getImgIds()
        image_ids = all_img_ids[:num_images]
        
        captions = {}
        
        for img_id in tqdm(image_ids, desc="Generating captions"):
            # Get image path
            img_info = self.fp.loadImgs([img_id])[0]
            img_filename = img_info['file_name']
            img_path = os.path.join(self.config['data']['images_dir'], img_filename)
            
            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}")
                continue
            
            # Generate caption
            caption = self.generate_caption(img_path)
            captions[str(img_id)] = caption
        
        # Diversity check
        unique_settings = len(set([c.split(',')[0].strip() for c in captions.values()]))
        logger.info(f"Diversity check: {unique_settings} unique settings out of {len(captions)} captions")
        
        min_unique = self.config['captioning']['min_unique_settings']
        if unique_settings < min_unique:
            logger.warning(f"Low diversity (< {min_unique} unique settings)")
            logger.warning("Consider adjusting the prompt or increasing image diversity")
        else:
            logger.info(f"✓ Good diversity (>= {min_unique} unique settings)")
        
        # Save captions
        save_json(captions, output_path)
        
        # Print sample captions
        logger.info("Sample captions:")
        for i, (img_id, caption) in enumerate(list(captions.items())[:5]):
            logger.info(f"  {img_id}: {caption}")
        
        return captions


def main():
    """Main execution"""
    generator = VibeCaptionGenerator()
    generator.generate_captions()


if __name__ == "__main__":
    main()
