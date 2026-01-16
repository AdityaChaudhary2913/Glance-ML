#!/usr/bin/env python3
"""
Generate markdown table from evaluation results
"""

import json
import sys
import os

def format_table(results_file='evaluation_results.json'):
    """Format evaluation results as markdown table"""
    
    # Check multiple possible locations
    possible_paths = [
        results_file,
        os.path.join('retriever', results_file),
        os.path.join('..', results_file)
    ]
    
    found_file = None
    for path in possible_paths:
        if os.path.exists(path):
            found_file = path
            break
    
    if not found_file:
        print(f"Error: {results_file} not found in any of these locations:")
        for p in possible_paths:
            print(f"  - {os.path.abspath(p)}")
        print("\nRun ./run_evaluation.sh first to generate results.")
        sys.exit(1)
    
    try:
        with open(found_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {found_file}: {e}")
        sys.exit(1)
    
    comparison = data.get('comparison', {})
    table_data = comparison.get('comparison_table', [])
    
    if not table_data:
        print("Error: No comparison data found in results file.")
        sys.exit(1)
    
    # Print markdown table
    print("\n## Performance\n")
    print("**Target**: 15-20% improvement over vanilla CLIP on compositional queries\n")
    print("| Query Type | Triple-Stream P@10 | Vanilla CLIP P@10 | Improvement |")
    print("|------------|-------------------|-------------------|-------------|")
    
    for row in table_data:
        query_type = row['query_type']
        triple_p10 = row['triple_p10']
        clip_p10 = row['clip_p10']
        improvement = row['improvement']
        
        if triple_p10 is not None:
            # Format improvement with proper sign
            imp_str = f"+{improvement:.1f}%" if improvement >= 0 else f"{improvement:.1f}%"
            print(f"| {query_type:<20} | {triple_p10:.2f} | {clip_p10:.2f} | {imp_str} |")
        else:
            print(f"| {query_type:<20} | TBD | TBD | TBD |")
    
    print()
    
    # Print summary
    avg_improvement = comparison.get('avg_improvement')
    if avg_improvement is not None:
        imp_str = f"+{avg_improvement:.1f}%" if avg_improvement >= 0 else f"{avg_improvement:.1f}%"
        print(f"\n**Average Improvement**: {imp_str}")
        
        if avg_improvement >= 15:
            print("✅ **Target achieved!** The triple-stream architecture outperforms vanilla CLIP.")
        elif avg_improvement >= 0:
            print("⚠️ Below target but positive. Consider tuning weight presets or improving V_fact/V_vibe quality.")
        else:
            print("❌ Negative improvement detected. This suggests an issue with the evaluation methodology.")
            print("   Consider manual evaluation or checking auto-judge thresholds.")
    
    print("\n---\n")
    print("**Methodology:**")
    print("- Automatic relevance judgment using keyword matching (30% threshold)")
    print("- Precision@10 (P@10): Percentage of relevant results in top 10")
    print("- Triple-Stream: Uses α·S_fact + β·S_vibe + γ·S_img fusion")
    print("- Vanilla CLIP: Standard CLIP image-text matching (visual only)")
    print()

if __name__ == '__main__':
    format_table()
