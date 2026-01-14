#!/usr/bin/env python3
"""
Quick start script to test the system setup
"""

import sys
import os

def check_imports():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        ('yaml', 'PyYAML'),
        ('PIL', 'Pillow'),
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('tqdm', 'tqdm'),
        ('chromadb', 'chromadb'),
        ('sentence_transformers', 'sentence-transformers'),
        ('transformers', 'transformers'),
    ]
    
    missing = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - MISSING")
            missing.append(package_name)
    
    return missing

def check_data():
    """Check if data files exist"""
    print("\nChecking data files...")
    
    data_files = [
        'data/instances_attributes_val2020.json',
        'data/attributes_val2020.json',
        'data/train',
        'fashionpedia-api-master/fashionpedia/fp.py'
    ]
    
    missing = []
    for filepath in data_files:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} - MISSING")
            missing.append(filepath)
    
    return missing

def check_config():
    """Check if config file exists"""
    print("\nChecking configuration...")
    
    if os.path.exists('config.yaml'):
        print("  ✓ config.yaml")
        return True
    else:
        print("  ✗ config.yaml - MISSING")
        return False

def main():
    print("="*60)
    print("Triple-Stream Fashion Search Engine")
    print("System Check")
    print("="*60 + "\n")
    
    # Check imports
    missing_packages = check_imports()
    
    # Check data
    missing_data = check_data()
    
    # Check config
    has_config = check_config()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not missing_packages and not missing_data and has_config:
        print("✓ All checks passed! System is ready.")
        print("\nNext steps:")
        print("  1. python caption_generator.py  # Generate vibe captions")
        print("  2. python indexer.py            # Build vector index")
        print("  3. python retriever.py          # Search images")
        print("  4. python evaluate.py           # Run evaluation")
        return 0
    else:
        print("✗ Some checks failed:")
        
        if missing_packages:
            print(f"\n  Missing packages: {', '.join(missing_packages)}")
            print("  Run: pip install -r requirements.txt")
        
        if missing_data:
            print(f"\n  Missing data files:")
            for f in missing_data:
                print(f"    - {f}")
        
        if not has_config:
            print("\n  Missing config.yaml")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
