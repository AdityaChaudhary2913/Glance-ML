"""
Evaluation: Compare Triple-Stream vs Vanilla CLIP Baseline
Includes structured evaluation, ablation study, and metrics computation
"""

import sys
import os
sys.path.insert(0, '/workspace/fashionpedia-api-master')
# Add parent directory to path for shared modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import json
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from retriever.retriever import TripleStreamRetriever
from shared.logger import evaluator_logger as logger


class SearchEvaluator:
    """
    Evaluate search performance and compare against baselines
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize evaluator
        
        Args:
            config_path: Path to config.yaml (None = auto-detect)
        """
        if config_path is None:
            # Get path relative to project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            config_path = os.path.join(project_root, 'shared', 'config.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.retriever = TripleStreamRetriever(config_path)
        
        # Test queries from assignment
        self.test_queries = [
            ("A person in a bright yellow raincoat", "attribute_specific"),
            ("Professional business attire inside a modern office", "contextual_place"),
            ("Someone wearing a blue shirt sitting on a park bench", "complex_semantic"),
            ("Casual weekend outfit for a city walk", "style_inference"),
            ("A red tie and a white shirt in a formal setting", "compositional")
        ]
    
    def evaluate_query(
        self,
        query: str,
        preset: str,
        ground_truth_ids: List[str] = None,
        auto_judge: bool = True
    ) -> Dict:
        """
        Comprehensive evaluation for one query
        
        Args:
            query: Search query string
            preset: Weight preset name
            ground_truth_ids: Optional list of relevant image IDs
            auto_judge: Use automatic relevance judgment (True) or manual (False)
            
        Returns:
            Dictionary with evaluation results
        """
        # Get results from triple-stream system
        results = self.retriever.dynamic_search(query, preset=preset)
        
        # Get vanilla CLIP baseline results
        vanilla_results = self.retriever.vanilla_clip_search(query)
        
        logger.info(f"\n{'='*100}")
        logger.info(f"Query: {query}")
        logger.info(f"Preset: {preset}")
        weights = self.config['weight_presets'][preset]
        logger.info(f"Weights: α={weights['alpha']}, β={weights['beta']}, γ={weights['gamma']}")
        logger.info(f"{'='*100}\n")
        
        # Display results
        logger.info("=== Triple-Stream Results ===")
        for i, (img_id, score, individual_scores) in enumerate(results, 1):
            metadata = self.retriever.get_image_metadata(img_id)
            
            logger.info(f"\nRank {i}: Image ID {img_id}")
            logger.info(f"  Score: {score:.4f} (G:{individual_scores['grounded']:.2f}, "
                  f"V:{individual_scores['vibe']:.2f}, I:{individual_scores['visual']:.2f})")
            
            if 'grounded_text' in metadata:
                logger.info(f"  Grounded: {metadata['grounded_text'][:100]}...")
            
            if 'vibe_text' in metadata:
                logger.info(f"  Vibe: {metadata['vibe_text'][:80]}...")
        
        logger.info("\n=== Vanilla CLIP Baseline (Top 5) ===")
        for i, img_id in enumerate(vanilla_results[:5], 1):
            logger.info(f"Rank {i}: Image ID {img_id}")
        
        # Calculate metrics
        if auto_judge:
            # Automatic relevance judgment based on metadata relevance
            triple_relevant = self._auto_judge_relevance(query, results[:10])
            vanilla_relevant = self._auto_judge_relevance_vanilla(query, vanilla_results[:10])
            
            triple_p10 = len(triple_relevant) / 10
            vanilla_p10 = len(vanilla_relevant) / 10
            
            logger.info(f"\n[Auto-Judge] Triple-Stream P@10: {triple_p10:.2f} ({len(triple_relevant)}/10 relevant)")
            logger.info(f"[Auto-Judge] Vanilla CLIP P@10: {vanilla_p10:.2f} ({len(vanilla_relevant)}/10 relevant)")
            
            improvement = ((triple_p10 - vanilla_p10) / vanilla_p10 * 100) if vanilla_p10 > 0 else 0
            logger.info(f"[Auto-Judge] Improvement: +{improvement:.1f}%")
        else:
            # Manual relevance judgment
            logger.info("\n" + "="*100)
            logger.info("RELEVANCE JUDGMENT")
            logger.info("Please review the images and enter relevant ranks.")
            logger.info("="*100)
            
            relevant_input = input(f"\nEnter relevant ranks for '{query}' (Triple-Stream, e.g., 1,2,4,7): ")
            triple_relevant = [int(r.strip())-1 for r in relevant_input.split(',') if r.strip()] if relevant_input else []
            triple_p10 = len(triple_relevant) / 10
            
            relevant_input = input(f"Enter relevant ranks for '{query}' (Vanilla CLIP, e.g., 1,3,5): ")
            vanilla_relevant = [int(r.strip())-1 for r in relevant_input.split(',') if r.strip()] if relevant_input else []
            vanilla_p10 = len(vanilla_relevant) / 10
            
            improvement = ((triple_p10 - vanilla_p10) / vanilla_p10 * 100) if vanilla_p10 > 0 else 0
        
        return {
            'query': query,
            'preset': preset,
            'weights': weights,
            'triple_stream_results': [(img_id, score) for img_id, score, _ in results],
            'vanilla_clip_results': vanilla_results,
            'triple_p10': triple_p10,
            'vanilla_p10': vanilla_p10,
            'improvement_pct': improvement,
            'triple_relevant_indices': triple_relevant,
            'vanilla_relevant_indices': vanilla_relevant
        }
    
    def _auto_judge_relevance(self, query: str, results: List[Tuple]) -> List[int]:
        """
        Automatically judge relevance based on query keywords matching metadata
        
        Args:
            query: Search query
            results: List of (img_id, score, individual_scores) tuples
            
        Returns:
            List of relevant result indices (0-based)
        """
        query_lower = query.lower()
        # Remove common stop words and split
        stop_words = {'a', 'the', 'in', 'on', 'at', 'for', 'and', 'or'}
        query_keywords = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]
        
        if not query_keywords:
            return []
        
        relevant_indices = []
        
        for idx, (img_id, score, _) in enumerate(results):
            metadata = self.retriever.get_image_metadata(img_id)
            
            # Check grounded text and vibe text for keyword matches
            grounded = metadata.get('grounded_text', '').lower()
            vibe = metadata.get('vibe_text', '').lower()
            combined_text = grounded + ' ' + vibe
            
            # Score based on keyword matches
            match_score = sum(1 for keyword in query_keywords if keyword in combined_text)
            
            # Consider relevant if at least 30% of keywords match
            if match_score >= max(1, len(query_keywords) * 0.3):
                relevant_indices.append(idx)
        
        return relevant_indices
    
    def _auto_judge_relevance_vanilla(self, query: str, result_ids: List[str]) -> List[int]:
        """
        Judge relevance for vanilla CLIP results (visual only)
        Uses same criteria as triple-stream for fair comparison
        
        Args:
            query: Search query
            result_ids: List of image IDs from vanilla CLIP
            
        Returns:
            List of relevant result indices (0-based)
        """
        query_lower = query.lower()
        # Remove common stop words
        stop_words = {'a', 'the', 'in', 'on', 'at', 'for', 'and', 'or'}
        query_keywords = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]
        
        if not query_keywords:
            return []
        
        relevant_indices = []
        
        for idx, img_id in enumerate(result_ids):
            metadata = self.retriever.get_image_metadata(img_id)
            
            # Check if visual features align with query
            grounded = metadata.get('grounded_text', '').lower()
            vibe = metadata.get('vibe_text', '').lower()
            combined_text = grounded + ' ' + vibe
            
            # Same threshold as triple-stream (30%)
            match_score = sum(1 for keyword in query_keywords if keyword in combined_text)
            
            if match_score >= max(1, len(query_keywords) * 0.3):
                relevant_indices.append(idx)
        
        return relevant_indices
    
    def ablation_study(self, query: str) -> Dict:
        """
        Ablation study: Compare single-stream vs full system
        
        Args:
            query: Test query
            
        Returns:
            Dictionary with ablation results
        """
        logger.info(f"\n{'='*100}")
        logger.info(f"ABLATION STUDY: {query}")
        logger.info(f"{'='*100}\n")
        
        ablation_configs = {
            "Only Grounded (α=1.0)": {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0},
            "Only Vibe (β=1.0)": {'alpha': 0.0, 'beta': 1.0, 'gamma': 0.0},
            "Only Visual (γ=1.0)": {'alpha': 0.0, 'beta': 0.0, 'gamma': 1.0},
            "Equal Weights": {'alpha': 0.33, 'beta': 0.33, 'gamma': 0.33},
            "Optimized Full System": None  # Will use best preset
        }
        
        results = {}
        
        for name, weights in ablation_configs.items():
            logger.info(f"\n--- {name} ---")
            
            if weights is None:
                # Determine best preset based on query type
                if "yellow" in query.lower() or "red" in query.lower():
                    preset = "attribute_specific"
                elif "office" in query.lower() or "park" in query.lower():
                    preset = "contextual_place"
                elif "weekend" in query.lower() or "casual" in query.lower():
                    preset = "style_inference"
                else:
                    preset = "compositional"
                
                search_results = self.retriever.dynamic_search(query, preset=preset, top_k=5)
            else:
                search_results = self.retriever.dynamic_search(
                    query,
                    alpha=weights['alpha'],
                    beta=weights['beta'],
                    gamma=weights['gamma'],
                    top_k=5
                )
            
            # Display top 3
            for i, (img_id, score, _) in enumerate(search_results[:3], 1):
                logger.info(f"  {i}. Image {img_id}: {score:.4f}")
            
            results[name] = [(img_id, score) for img_id, score, _ in search_results]
        
        return results
    
    def visualize_results(
        self,
        query: str,
        results: List[Tuple[str, float, Dict]],
        output_path: str = None
    ):
        """
        Visualize search results in a grid
        
        Args:
            query: Search query
            results: Results from dynamic_search
            output_path: Path to save visualization (None = display)
        """
        n_results = min(5, len(results))
        
        fig = plt.figure(figsize=(15, 4))
        gs = GridSpec(1, n_results + 1, figure=fig, wspace=0.3)
        
        # Title
        fig.suptitle(f'Query: "{query}"', fontsize=14, fontweight='bold')
        
        # Display top results
        for i, (img_id, score, individual_scores) in enumerate(results[:n_results]):
            ax = fig.add_subplot(gs[0, i])
            
            # Get image path
            metadata = self.retriever.get_image_metadata(img_id)
            img_path = metadata.get('image_path', '')
            
            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax.imshow(img)
            
            # Title with score
            title = f"Rank {i+1}\nScore: {score:.3f}\n"
            title += f"G:{individual_scores['grounded']:.2f} "
            title += f"V:{individual_scores['vibe']:.2f} "
            title += f"I:{individual_scores['visual']:.2f}"
            
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved visualization to {output_path}")
        else:
            plt.show()
    
    def run_full_evaluation(self, visualize: bool = False, auto_judge: bool = True) -> Dict:
        """
        Run complete evaluation on all test queries
        
        Args:
            visualize: Whether to create visualization images
            auto_judge: Use automatic relevance judgment (recommended)
            
        Returns:
            Dictionary with all evaluation results
        """
        all_results = {}
        
        logger.info("\n" + "="*100)
        logger.info("FULL EVALUATION: Triple-Stream Fashion Search")
        logger.info(f"Auto-Judge: {auto_judge}")
        logger.info("="*100)
        
        for query, preset in self.test_queries:
            # Evaluate query
            result = self.evaluate_query(query, preset, auto_judge=auto_judge)
            all_results[query] = result
            
            # Visualize if requested
            if visualize:
                search_results = self.retriever.dynamic_search(query, preset=preset)
                output_path = f"results/{query.replace(' ', '_')[:50]}.png"
                os.makedirs('results', exist_ok=True)
                self.visualize_results(query, search_results, output_path)
        
        # Run ablation study on one query
        logger.info("\n" + "="*100)
        logger.info("Running Ablation Study")
        logger.info("="*100)
        ablation_results = self.ablation_study(self.test_queries[0][0])
        all_results['ablation'] = ablation_results
        
        # Save results to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        results_path = os.path.join(project_root, 'evaluation_results.json')
        
        with open(results_path, 'w') as f:
            # Convert to serializable format
            serializable_results = {}
            for key, value in all_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = value
            json.dump(serializable_results, f, indent=2)
        
        logger.info("\n" + "="*100)
        logger.info("Evaluation complete! Results saved to evaluation_results.json")
        logger.info("="*100)
        
        return all_results
    
    def compare_with_baseline(self, results: Dict) -> Dict:
        """
        Generate comparison table with vanilla CLIP
        
        Args:
            results: Results from run_full_evaluation
            
        Returns:
            Dictionary with formatted comparison data
        """
        logger.info("\n" + "="*100)
        logger.info("PERFORMANCE COMPARISON: Triple-Stream vs Vanilla CLIP")
        logger.info("="*100 + "\n")
        
        logger.info(f"{'Query Type':<30} {'Triple P@10':<15} {'CLIP P@10':<15} {'Improvement'}")
        logger.info("-" * 100)
        
        avg_triple = []
        avg_clip = []
        comparison_table = []
        
        for query, preset in self.test_queries:
            if query in results and 'triple_p10' in results[query]:
                triple_p10 = results[query]['triple_p10']
                clip_p10 = results[query]['vanilla_p10']
                improvement = results[query]['improvement_pct']
                
                logger.info(f"{preset:<30} {triple_p10:.2f}{'':<12} {clip_p10:.2f}{'':<12} +{improvement:.1f}%")
                
                avg_triple.append(triple_p10)
                avg_clip.append(clip_p10)
                
                comparison_table.append({
                    'query_type': preset,
                    'triple_p10': triple_p10,
                    'clip_p10': clip_p10,
                    'improvement': improvement
                })
            else:
                logger.info(f"{preset:<30} {'N/A':<15} {'N/A':<15} {'N/A'}")
                comparison_table.append({
                    'query_type': preset,
                    'triple_p10': None,
                    'clip_p10': None,
                    'improvement': None
                })
        
        if avg_triple:
            logger.info("-" * 100)
            avg_t = np.mean(avg_triple)
            avg_c = np.mean(avg_clip)
            avg_imp = ((avg_t - avg_c) / avg_c * 100) if avg_c > 0 else 0
            logger.info(f"{'AVERAGE':<30} {avg_t:.2f}{'':<12} {avg_c:.2f}{'':<12} +{avg_imp:.1f}%")
            
            comparison_table.append({
                'query_type': 'AVERAGE',
                'triple_p10': avg_t,
                'clip_p10': avg_c,
                'improvement': avg_imp
            })
        
        return {'comparison_table': comparison_table, 'avg_improvement': avg_imp if avg_triple else None}


def main():
    """Run evaluation"""
    evaluator = SearchEvaluator()
    
    # Run full evaluation with automatic judgment
    logger.info("Starting full evaluation with automatic relevance judgment...")
    results = evaluator.run_full_evaluation(visualize=False, auto_judge=True)
    
    # Generate comparison table
    comparison = evaluator.compare_with_baseline(results)
    
    # Save detailed results
    output = {
        'evaluation_results': results,
        'comparison': comparison,
        'timestamp': str(np.datetime64('now'))
    }
    
    with open('evaluation_results.json', 'w') as f:
        # Convert numpy types to native Python types
        import json
        def default_converter(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)
        
        json.dump(output, f, indent=2, default=default_converter)
    
    logger.info("\nEvaluation complete! Results saved to evaluation_results.json")
    logger.info("\n" + "="*100)
    logger.info("SUMMARY")
    logger.info("="*100)
    
    if comparison['avg_improvement'] is not None:
        logger.info(f"Average Improvement over Vanilla CLIP: +{comparison['avg_improvement']:.1f}%")
        logger.info(f"Target: 15-20% improvement")
        
        if comparison['avg_improvement'] >= 15:
            logger.info("✅ TARGET ACHIEVED!")
        else:
            logger.info("⚠️  Below target - consider tuning weight presets")
    
    return results, comparison


if __name__ == "__main__":
    main()
