"""
Weight Optimization using Bayesian Optimization (Optuna)
Automatically finds optimal alpha, beta, gamma weights for different query presets
"""

import sys
import os
sys.path.insert(0, '/workspace/fashionpedia-api-master')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import json
import optuna
from optuna.samplers import TPESampler
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

from retriever.retriever import TripleStreamRetriever
from shared.logger import retriever_logger as logger


class WeightOptimizer:
    """
    Bayesian optimization for finding optimal fusion weights
    """
    
    def __init__(self, config_path: str = None):
        """Initialize optimizer with retriever"""
        if config_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            config_path = os.path.join(project_root, 'shared', 'config.yaml')
        else:
            project_root = os.path.abspath(os.path.join(os.path.dirname(config_path), '..'))
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_path = config_path
        self.project_root = project_root
        self.retriever = TripleStreamRetriever(config_path)
        
        # Test queries with diverse characteristics
        self.test_queries = [
            # Attribute-specific queries
            "A person in a bright yellow raincoat",
            "Someone wearing a red dress",
            "White shirt and blue jeans",
            
            # Contextual/place queries
            "Professional business attire inside a modern office",
            "Beachwear at the seaside",
            "Winter clothes in snowy mountains",
            
            # Complex semantic queries
            "Someone wearing a blue shirt sitting on a park bench",
            "A person reading a book in a cafe",
            "Walking a dog in the park",
            
            # Style inference queries
            "Casual weekend outfit for a city walk",
            "Elegant evening gown for a formal event",
            "Sporty athleisure for gym workout",
            
            # Compositional queries
            "A red tie and a white shirt in a formal setting",
            "Black leather jacket with ripped jeans",
            "Floral dress with sun hat"
        ]
    
    def objective_global(self, trial: optuna.Trial) -> float:
        """
        Objective function for global weight optimization
        Optimizes a single set of weights for all query types
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Negative mean average precision (we minimize, so negate to maximize)
        """
        # Sample weights (they will be normalized to sum to 1.0)
        alpha = trial.suggest_float('alpha', 0.0, 1.0)
        beta = trial.suggest_float('beta', 0.0, 1.0)
        gamma = trial.suggest_float('gamma', 0.0, 1.0)
        
        # Normalize weights
        total = alpha + beta + gamma
        if total > 0:
            alpha, beta, gamma = alpha/total, beta/total, gamma/total
        else:
            return -999.0  # Invalid weights
        
        # Evaluate on all test queries
        total_score = 0.0
        
        for query in self.test_queries:
            # Get search results
            results = self.retriever.dynamic_search(
                query,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                top_k=10,
                expand=True
            )
            
            # Compute relevance score (auto-judge based on metadata matching)
            score = self._compute_relevance_score(query, results)
            total_score += score
        
        mean_score = total_score / len(self.test_queries)
        
        logger.info(f"Trial {trial.number}: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f} → Score={mean_score:.4f}")
        
        return mean_score
    
    def objective_per_preset(self, trial: optuna.Trial, query_type: str, queries: List[str]) -> float:
        """
        Objective function for preset-specific weight optimization
        Optimizes weights for a specific query type/preset
        
        Args:
            trial: Optuna trial object
            query_type: Type of queries (e.g., 'attribute_specific', 'contextual_place')
            queries: List of test queries for this type
            
        Returns:
            Mean relevance score across queries
        """
        # Sample weights
        alpha = trial.suggest_float('alpha', 0.0, 1.0)
        beta = trial.suggest_float('beta', 0.0, 1.0)
        gamma = trial.suggest_float('gamma', 0.0, 1.0)
        
        # Normalize weights
        total = alpha + beta + gamma
        if total > 0:
            alpha, beta, gamma = alpha/total, beta/total, gamma/total
        else:
            return -999.0
        
        # Evaluate on queries of this type
        total_score = 0.0
        
        for query in queries:
            results = self.retriever.dynamic_search(
                query,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                top_k=10,
                expand=True
            )
            
            score = self._compute_relevance_score(query, results)
            total_score += score
        
        mean_score = total_score / len(queries)
        
        logger.info(f"[{query_type}] Trial {trial.number}: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f} → Score={mean_score:.4f}")
        
        return mean_score
    
    def _compute_relevance_score(self, query: str, results: List[Tuple], use_semantic: bool = False) -> float:
        """
        Compute relevance score using semantic similarity or keyword matching
        Combination of precision and NDCG
        
        Args:
            query: Search query
            results: List of (img_id, score, individual_scores) tuples
            use_semantic: Use CLIP text embeddings for semantic similarity (default: True)
            
        Returns:
            Relevance score (higher is better)
        """
        if not results:
            return 0.0
        
        relevance_scores = []
        
        if use_semantic:
            # SEMANTIC SIMILARITY METHOD (Better for abstract queries)
            query_embedding = self.retriever.clip_model.encode(query, convert_to_numpy=True)
            
            for img_id, final_score, _ in results:
                metadata = self.retriever.get_image_metadata(img_id)
                
                grounded = metadata.get('grounded_text', '')
                vibe = metadata.get('vibe_text', '')
                combined_text = f"{grounded} {vibe}".strip()
                
                if not combined_text:
                    relevance_scores.append(0.0)
                    continue
                
                text_embedding = self.retriever.clip_model.encode(combined_text, convert_to_numpy=True)
                
                # Cosine similarity
                similarity = np.dot(query_embedding, text_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
                )
                
                # Clip to [0, 1] and use as relevance score
                relevance = max(0.0, min(1.0, similarity))
                relevance_scores.append(relevance)
        
        else:
            # KEYWORD MATCHING METHOD (Original)
            query_lower = query.lower()
            stop_words = {'a', 'the', 'in', 'on', 'at', 'for', 'and', 'or', 'with'}
            query_keywords = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]
            
            if not query_keywords:
                return 0.0
            
            for img_id, final_score, _ in results:
                metadata = self.retriever.get_image_metadata(img_id)
                
                grounded = metadata.get('grounded_text', '').lower()
                vibe = metadata.get('vibe_text', '').lower()
                combined_text = grounded + ' ' + vibe
                
                # Count keyword matches
                match_count = sum(1 for keyword in query_keywords if keyword in combined_text)
                
                # Relevance is percentage of keywords matched
                relevance = match_count / len(query_keywords)
                relevance_scores.append(relevance)
        
        # Compute metrics
        # 1. Precision@10: Fraction of results with relevance >= 0.3
        relevant_mask = np.array(relevance_scores) >= 0.3
        precision_at_10 = relevant_mask.sum() / len(results)
        
        # 2. NDCG@10: Normalized Discounted Cumulative Gain
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_scores))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # 3. Mean Reciprocal Rank: Position of first highly relevant result
        first_relevant_idx = next((idx for idx, rel in enumerate(relevance_scores) if rel >= 0.5), None)
        mrr = 1.0 / (first_relevant_idx + 1) if first_relevant_idx is not None else 0.0
        
        # Combined score: weighted average
        combined_score = 0.4 * precision_at_10 + 0.4 * ndcg + 0.2 * mrr
        
        return combined_score
    
    def optimize_global_weights(self, n_trials: int = 50) -> Dict:
        """
        Optimize a single set of weights for all query types
        
        Args:
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with optimal weights and performance
        """
        logger.info("="*100)
        logger.info("GLOBAL WEIGHT OPTIMIZATION")
        logger.info(f"Optimizing single weight set for {len(self.test_queries)} diverse queries")
        logger.info(f"Running {n_trials} trials with Bayesian optimization")
        logger.info("="*100)
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            study_name='global_weights'
        )
        
        # Optimize
        study.optimize(self.objective_global, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Normalize best weights
        alpha = best_params['alpha']
        beta = best_params['beta']
        gamma = best_params['gamma']
        total = alpha + beta + gamma
        alpha, beta, gamma = alpha/total, beta/total, gamma/total
        
        result = {
            'alpha': round(alpha, 3),
            'beta': round(beta, 3),
            'gamma': round(gamma, 3),
            'score': round(best_value, 4),
            'n_trials': n_trials
        }
        
        logger.info("\n" + "="*100)
        logger.info("GLOBAL OPTIMIZATION COMPLETE")
        logger.info(f"Best weights: α={result['alpha']}, β={result['beta']}, γ={result['gamma']}")
        logger.info(f"Best score: {result['score']}")
        logger.info("="*100 + "\n")
        
        return result
    
    def optimize_per_preset(self, n_trials: int = 30) -> Dict[str, Dict]:
        """
        Optimize weights separately for each query preset/type
        
        Args:
            n_trials: Number of trials per preset
            
        Returns:
            Dictionary mapping preset names to optimal weights
        """
        logger.info("="*100)
        logger.info("PER-PRESET WEIGHT OPTIMIZATION")
        logger.info(f"Optimizing weights for each query type separately")
        logger.info(f"Running {n_trials} trials per preset")
        logger.info("="*100)
        
        # Define query groups by type
        query_groups = {
            'attribute_specific': [
                "A person in a bright yellow raincoat",
                "Someone wearing a red dress",
                "White shirt and blue jeans"
            ],
            'contextual_place': [
                "Professional business attire inside a modern office",
                "Beachwear at the seaside",
                "Winter clothes in snowy mountains"
            ],
            'complex_semantic': [
                "Someone wearing a blue shirt sitting on a park bench",
                "A person reading a book in a cafe",
                "Walking a dog in the park"
            ],
            'style_inference': [
                "Casual weekend outfit for a city walk",
                "Elegant evening gown for a formal event",
                "Sporty athleisure for gym workout"
            ],
            'compositional': [
                "A red tie and a white shirt in a formal setting",
                "Black leather jacket with ripped jeans",
                "Floral dress with sun hat"
            ]
        }
        
        results = {}
        
        for preset_name, queries in query_groups.items():
            logger.info(f"\n{'='*100}")
            logger.info(f"Optimizing: {preset_name}")
            logger.info(f"Test queries: {len(queries)}")
            logger.info(f"{'='*100}\n")
            
            # Create study for this preset
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                study_name=f'{preset_name}_weights'
            )
            
            # Optimize with lambda to pass query type and queries
            study.optimize(
                lambda trial: self.objective_per_preset(trial, preset_name, queries),
                n_trials=n_trials,
                show_progress_bar=True
            )
            
            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Normalize
            alpha = best_params['alpha']
            beta = best_params['beta']
            gamma = best_params['gamma']
            total = alpha + beta + gamma
            alpha, beta, gamma = alpha/total, beta/total, gamma/total
            
            results[preset_name] = {
                'alpha': round(alpha, 3),
                'beta': round(beta, 3),
                'gamma': round(gamma, 3),
                'score': round(best_value, 4),
                'n_trials': n_trials
            }
            
            logger.info(f"\n[{preset_name}] Best weights: α={results[preset_name]['alpha']}, "
                       f"β={results[preset_name]['beta']}, γ={results[preset_name]['gamma']}")
            logger.info(f"[{preset_name}] Best score: {results[preset_name]['score']}\n")
        
        logger.info("\n" + "="*100)
        logger.info("PER-PRESET OPTIMIZATION COMPLETE")
        logger.info("="*100 + "\n")
        
        return results
    
    def save_optimized_weights(self, global_weights: Dict = None, preset_weights: Dict[str, Dict] = None):
        """
        Save optimized weights to config file
        
        Args:
            global_weights: Optimized global weights
            preset_weights: Optimized per-preset weights
        """
        # Load current config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Backup original config
        backup_path = self.config_path.replace('.yaml', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml')
        with open(backup_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Backup saved to: {backup_path}")
        
        # Add optimized weights section
        if 'optimized_weights' not in config:
            config['optimized_weights'] = {}
        
        if global_weights:
            config['optimized_weights']['global'] = global_weights
            logger.info(f"Saved global optimized weights: α={global_weights['alpha']}, "
                       f"β={global_weights['beta']}, γ={global_weights['gamma']}")
        
        if preset_weights:
            config['optimized_weights']['presets'] = preset_weights
            logger.info("Saved optimized weights for all presets")
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Updated config saved to: {self.config_path}")
    
    def run_full_optimization(self, global_trials: int = 50, preset_trials: int = 30):
        """
        Run complete optimization workflow
        
        Args:
            global_trials: Number of trials for global optimization
            preset_trials: Number of trials per preset
        """
        logger.info("\n" + "="*100)
        logger.info("FULL WEIGHT OPTIMIZATION WORKFLOW")
        logger.info(f"Global trials: {global_trials}, Per-preset trials: {preset_trials}")
        logger.info("="*100 + "\n")
        
        # 1. Optimize global weights
        logger.info("\n### STEP 1: Global Weight Optimization ###\n")
        global_weights = self.optimize_global_weights(n_trials=global_trials)
        
        # 2. Optimize per-preset weights
        logger.info("\n### STEP 2: Per-Preset Weight Optimization ###\n")
        preset_weights = self.optimize_per_preset(n_trials=preset_trials)
        
        # 3. Save results
        logger.info("\n### STEP 3: Saving Results ###\n")
        self.save_optimized_weights(global_weights=global_weights, preset_weights=preset_weights)
        
        # 4. Generate summary report
        self._generate_report(global_weights, preset_weights)
        
        logger.info("\n" + "="*100)
        logger.info("OPTIMIZATION COMPLETE!")
        logger.info("="*100 + "\n")
    
    def _generate_report(self, global_weights: Dict, preset_weights: Dict[str, Dict]):
        """Generate optimization report"""
        report_path = os.path.join(self.project_root, 'outputs', 'weight_optimization_report.json')
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'global_weights': global_weights,
            'preset_weights': preset_weights,
            'comparison': {
                'original_vs_optimized': {}
            }
        }
        
        # Compare with original weights from config
        for preset_name, opt_weights in preset_weights.items():
            if preset_name in self.config['weight_presets']:
                orig = self.config['weight_presets'][preset_name]
                report['comparison']['original_vs_optimized'][preset_name] = {
                    'original': orig,
                    'optimized': {
                        'alpha': opt_weights['alpha'],
                        'beta': opt_weights['beta'],
                        'gamma': opt_weights['gamma']
                    },
                    'score_improvement': opt_weights.get('score', 0.0)
                }
        
        # Save report
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Optimization report saved to: {report_path}")


def main():
    """Main optimization workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize fusion weights using Bayesian optimization')
    parser.add_argument('--mode', choices=['global', 'preset', 'full'], default='full',
                       help='Optimization mode: global (single weights), preset (per-type), or full (both)')
    parser.add_argument('--global-trials', type=int, default=50,
                       help='Number of trials for global optimization')
    parser.add_argument('--preset-trials', type=int, default=30,
                       help='Number of trials per preset optimization')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = WeightOptimizer()
    
    # Run optimization based on mode
    if args.mode == 'global':
        global_weights = optimizer.optimize_global_weights(n_trials=args.global_trials)
        optimizer.save_optimized_weights(global_weights=global_weights)
    elif args.mode == 'preset':
        preset_weights = optimizer.optimize_per_preset(n_trials=args.preset_trials)
        optimizer.save_optimized_weights(preset_weights=preset_weights)
    else:  # full
        optimizer.run_full_optimization(
            global_trials=args.global_trials,
            preset_trials=args.preset_trials
        )


if __name__ == '__main__':
    main()
