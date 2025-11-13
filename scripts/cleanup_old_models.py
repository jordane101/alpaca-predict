#!/usr/bin/env python3
"""
Script to clean up old HMM model files that were trained with 3 components.
Since we've switched to 2 components, the old models are incompatible.

Author - Eli Jordan
Date - 10/17/2025
"""

import os
from pathlib import Path
import re

def cleanup_old_models():
    """Remove model files with 3 components (pattern: *_3_*.pkl and *_3_*.json)"""
    model_dir = Path("hmm_models")
    
    if not model_dir.exists():
        print(f"Model directory {model_dir} does not exist.")
        return
    
    # Pattern to match files with 3 components: ticker_3_order.pkl or ticker_3_order.json
    pattern = re.compile(r'.*_3_\d+\.(pkl|json)$')
    
    deleted_count = 0
    for file_path in model_dir.iterdir():
        if file_path.is_file() and pattern.match(file_path.name):
            try:
                file_path.unlink()
                print(f"Deleted: {file_path.name}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file_path.name}: {e}")
    
    print(f"\nTotal files deleted: {deleted_count}")
    print("Old 3-component models have been cleaned up.")
    print("New 2-component models will be created automatically when needed.")

if __name__ == "__main__":
    print("=" * 60)
    print("HMM Model Cleanup - Removing 3-component models")
    print("=" * 60)
    
    response = input("\nThis will delete all model files with 3 components. Continue? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        cleanup_old_models()
    else:
        print("Cleanup cancelled.")
