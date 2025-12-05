#!/usr/bin/env python3
"""Data preprocessing CLI"""

import argparse
from pathlib import Path
import sys
import os
import pandas as pd

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config, META_CONFIG
from kgraph.preprocessing import process_meme_data


def main():
    parser = argparse.ArgumentParser(description='Preprocess meme data')
    parser.add_argument('--input-dir', default='data/raw', help='Input directory')
    parser.add_argument('--output-dir', default='data/cleaned_data', help='Output directory')
    parser.add_argument('--index-key', help='Path to index key file')

    args = parser.parse_args()

if __name__ == "__main__":

    data_dir = Config.META_DATA_DIR / "raw"
    output_dir = Config.META_DATA_DIR / "cleaned_data"

    os.makedirs(output_dir, exist_ok=True)

    # Default files to process
    files_to_process = [
        data_dir / "all_memes_merged.csv"
    ]

    all_files = [f for f in files_to_process if f.exists()]

    # Path to index key
    index_key_path = data_dir / "complete_index.csv"

    print(f"Processing {len(all_files)} files: {[f.name for f in all_files]}")
    combined_output = output_dir / "cleaned_source_data.csv"

    try:
        all_memes = process_meme_data(
            [str(f) for f in all_files],  # Convert Path objects to strings
            str(index_key_path),
            combine=True,
            combined_output=str(combined_output)
        )

        # Print statistics
        if isinstance(all_memes, pd.DataFrame):
            print(f"\nCombined dataset has {len(all_memes)} rows")
            print("\nCounts of non-null values in key columns:")
            key_columns = ['Title', 'URI_Safe_Title', 'Part of a series on', 'URI_Safe_Part of a series on',
                           'Source_File']
            for col in key_columns:
                if col in all_memes.columns:
                    count = all_memes[col].notna().sum()
                    print(f"  {col}: {count} / {len(all_memes)}")

            # Print source file distribution
            if 'Source_File' in all_memes.columns:
                print("\nRows per source file:")
                source_counts = all_memes['Source_File'].value_counts()
                for source, count in source_counts.items():
                    print(f"  {source}: {count}")

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback

        traceback.print_exc()