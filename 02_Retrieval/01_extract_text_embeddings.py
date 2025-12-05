"""
Extract text embeddings from OCR data in CSV files using Sentence Transformers
Follows the same chunking pattern as extract_text_embeddings.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json
import pickle
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config.config import RETRIEVAL_CONFIG, CORPUS_CONFIG

# Use retrieval config values
TEXT_MODEL_NAME = RETRIEVAL_CONFIG.TEXT_MODEL_NAME
TEXT_EMBEDDING_DIM = RETRIEVAL_CONFIG.TEXT_EMBEDDING_DIM
TEXT_BATCH_SIZE = RETRIEVAL_CONFIG.TEXT_BATCH_SIZE
TEXT_CHUNK_SIZE = RETRIEVAL_CONFIG.TEXT_CHUNK_SIZE

# Text-specific paths
TEXT_EMBEDDINGS_DIR = RETRIEVAL_CONFIG.TEXT_EMBEDDINGS_DIR
TEXT_METADATA_DIR = RETRIEVAL_CONFIG.TEXT_METADATA_DIR
OCR_FILES = RETRIEVAL_CONFIG.OCR_FILES

SUPPORTED_EXTENSIONS = CORPUS_CONFIG.SUPPORTED_IMAGE_EXTENSIONS

class OCRTextDataset(Dataset):
    """Dataset for processing OCR text files"""

    def __init__(self, text_data):
        """
        Args:
            text_data: List of dictionaries with text information
        """
        self.text_data = text_data

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        return self.text_data[idx]


class MemeTextExtractor:
    """Extract embeddings from OCR text data in CSV files"""

    def __init__(self):
        print("Loading Sentence Transformer model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load sentence transformer model
        self.model = SentenceTransformer(TEXT_MODEL_NAME, device=str(self.device))

        # Set to eval mode
        self.model.eval()

        # Metadata tracking
        self.metadata = {
            'texts': [],
            'class_counts': defaultdict(int),
            'dataset_stats': {
                'confirmed': 0,
                'unconfirmed': 0,
                'empty_ocr': 0,
                'total': 0,
                'files_processed': []
            }
        }

        print(f"Model: {TEXT_MODEL_NAME}")
        print(f"Device: {self.device}")
        print(f"Embedding dimension: {TEXT_EMBEDDING_DIM}")

    def scan_ocr_csv_files(self):
        """Read OCR data from CSV files"""
        print("\nScanning OCR CSV files...")
        all_texts = []

        for csv_file_path in OCR_FILES:
            csv_path = Path(csv_file_path)

            if not csv_path.exists():
                print(f"Warning: CSV file not found: {csv_path}")
                continue

            print(f"\nProcessing: {csv_path}")
            print(f"  File size: {csv_path.stat().st_size / (1024 * 1024):.2f} MB")

            # Determine if this is confirmed or unconfirmed based on filename
            is_confirmed = 'unconfirmed' not in csv_path.name.lower()
            dataset_type = 'confirmed' if is_confirmed else 'unconfirmed'

            # Read CSV file
            try:
                # Try reading with different encodings if needed
                try:
                    df = pd.read_csv(csv_path, encoding='utf-8')
                except UnicodeDecodeError:
                    print("  Trying latin-1 encoding...")
                    df = pd.read_csv(csv_path, encoding='latin-1')

                print(f"  Loaded {len(df):,} rows")
                print(f"  Columns: {list(df.columns)}")

                # Detect OCR column - try common names
                ocr_column = None
                possible_ocr_columns = ['ocr', 'OCR', 'text', 'Text', 'ocr_text',
                                        'extracted_text', 'content', 'meme_text']

                for col in possible_ocr_columns:
                    if col in df.columns:
                        ocr_column = col
                        break

                if ocr_column is None:
                    # If no standard OCR column found, show available columns
                    print(f"  No standard OCR column found. Available columns:")
                    for col in df.columns:
                        sample_value = df[col].iloc[0] if len(df) > 0 else "N/A"
                        print(f"    - {col}: {str(sample_value)[:100]}")

                    # Try to guess which column contains text
                    text_columns = []
                    for col in df.columns:
                        if df[col].dtype == object:  # String columns
                            avg_length = df[col].fillna('').str.len().mean()
                            if avg_length > 10:  # Likely contains text, not just IDs
                                text_columns.append((col, avg_length))

                    if text_columns:
                        # Use the column with longest average text
                        ocr_column = max(text_columns, key=lambda x: x[1])[0]
                        print(f"  Auto-detected OCR column: '{ocr_column}' (avg length: {text_columns[0][1]:.1f})")

                if ocr_column is None:
                    print(f"  ERROR: Could not identify OCR column in {csv_path}")
                    continue

                print(f"  Using OCR column: '{ocr_column}'")

                # Process each row
                processed_count = 0
                empty_count = 0

                for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Processing rows"):
                    ocr_text = str(row[ocr_column]) if pd.notna(row[ocr_column]) else ""

                    # Skip empty or 'nan' texts
                    if not ocr_text or ocr_text.lower() in ['nan', 'none', 'null', '', '-'] or len(
                            ocr_text.strip()) < 2:
                        empty_count += 1
                        self.metadata['dataset_stats']['empty_ocr'] += 1
                        continue

                    # Get class_id from 'label' column (matches folder name)
                    class_id = None
                    if 'label' in df.columns and pd.notna(row['label']):
                        class_id = str(row['label'])
                    elif 'Label' in df.columns and pd.notna(row['Label']):
                        class_id = str(row['Label'])
                    else:
                        # Fallback: try to extract from filename if we have it
                        if 'file' in df.columns and pd.notna(row['file']):
                            filename = str(row['file'])
                            if '-' in filename:
                                class_id = filename.split('-')[0]
                        elif 'File' in df.columns and pd.notna(row['File']):
                            filename = str(row['File'])
                            if '-' in filename:
                                class_id = filename.split('-')[0]

                    if not class_id:
                        class_id = "unknown"

                    # Get filename for later image retrieval
                    filename = None
                    if 'file' in df.columns and pd.notna(row['file']):
                        filename = str(row['file'])
                    elif 'File' in df.columns and pd.notna(row['File']):
                        filename = str(row['File'])

                    text_entry = {
                        'text_id': f"row_{idx}",
                        'text': ocr_text,
                        'class_id': class_id,  # Now matches folder name!
                        'dataset_type': dataset_type,
                        'source_file': str(csv_path),
                        'csv_index': idx,
                        'text_length': len(ocr_text),
                        'word_count': len(ocr_text.split()),
                        'original_path': filename,  # Store filename here
                        'meta_file': filename  # Also store as meta_file for compatibility
                    }

                    all_texts.append(text_entry)
                    self.metadata['class_counts'][class_id] += 1
                    self.metadata['dataset_stats'][dataset_type] += 1
                    processed_count += 1

                print(f"  Processed: {processed_count:,} valid texts")
                print(f"  Skipped: {empty_count:,} empty/invalid texts")

                self.metadata['dataset_stats']['files_processed'].append(str(csv_path))

            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                continue

        self.metadata['dataset_stats']['total'] = len(all_texts)

        # Add popularity scores
        for text_entry in all_texts:
            text_entry['popularity'] = self.metadata['class_counts'][text_entry['class_id']]
            text_entry['popularity_rank'] = None

        # Calculate popularity ranks
        sorted_classes = sorted(self.metadata['class_counts'].items(),
                                key=lambda x: x[1], reverse=True)
        class_to_rank = {cls: rank for rank, (cls, _) in enumerate(sorted_classes)}

        for text_entry in all_texts:
            text_entry['popularity_rank'] = class_to_rank[text_entry['class_id']]

        # Print statistics
        print(f"\n{'=' * 60}")
        print("OCR Data Statistics:")
        print(f"  Files processed: {len(self.metadata['dataset_stats']['files_processed'])}")
        for file in self.metadata['dataset_stats']['files_processed']:
            print(f"    - {Path(file).name}")
        print(f"  Total valid texts: {len(all_texts):,}")
        print(f"  Empty/invalid texts skipped: {self.metadata['dataset_stats']['empty_ocr']:,}")
        print(f"  Unique classes: {len(self.metadata['class_counts']):,}")
        print(f"  Confirmed texts: {self.metadata['dataset_stats']['confirmed']:,}")
        print(f"  Unconfirmed texts: {self.metadata['dataset_stats']['unconfirmed']:,}")

        # Text statistics
        if all_texts:
            text_lengths = [t['text_length'] for t in all_texts]
            word_counts = [t['word_count'] for t in all_texts]
            print(f"\nText Statistics:")
            print(f"  Avg text length: {np.mean(text_lengths):.1f} chars")
            print(f"  Avg word count: {np.mean(word_counts):.1f} words")
            print(f"  Min/Max text length: {min(text_lengths)}/{max(text_lengths)} chars")
            print(f"  Min/Max word count: {min(word_counts)}/{max(word_counts)} words")

            # Top 5 classes by count
            print(f"\nTop 5 classes by text count:")
            for i, (cls, count) in enumerate(sorted_classes[:5]):
                print(f"  {i + 1}. {cls}: {count:,} texts")

        print(f"{'=' * 60}")

        return all_texts

    def extract_embeddings_batch(self, text_batch):
        """Extract embeddings for a batch of texts"""

        # Prepare texts for encoding
        texts_to_encode = []
        valid_indices = []

        for idx, text_entry in enumerate(text_batch):
            text = text_entry['text']

            # Truncate very long texts (sentence-transformers has max length)
            if len(text) > 5000:  # Characters, not tokens
                text = text[:5000]

            # Skip empty texts
            if text and text.strip():
                texts_to_encode.append(text)
                valid_indices.append(idx)

        if not texts_to_encode:
            return None, []

        # Encode texts using sentence transformer
        print(f"  Encoding {len(texts_to_encode)} texts...")

        # Sentence transformers handles batching internally
        with torch.no_grad():
            embeddings = self.model.encode(
                texts_to_encode,
                batch_size=TEXT_BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalize
            )

        return embeddings, valid_indices

    def process_in_chunks(self, all_texts):
        """Process texts in chunks to manage memory"""

        Path(TEXT_EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)
        Path(TEXT_METADATA_DIR).mkdir(parents=True, exist_ok=True)

        num_chunks = (len(all_texts) + TEXT_CHUNK_SIZE - 1) // TEXT_CHUNK_SIZE

        print(f"\nProcessing {len(all_texts)} texts in {num_chunks} chunks...")
        print(f"Chunk size: {TEXT_CHUNK_SIZE}")

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * TEXT_CHUNK_SIZE
            end_idx = min(start_idx + TEXT_CHUNK_SIZE, len(all_texts))
            chunk = all_texts[start_idx:end_idx]

            print(f"\n{'=' * 60}")
            print(f"Chunk {chunk_idx + 1}/{num_chunks}: Processing texts {start_idx:,}-{end_idx:,}")

            # Extract embeddings
            embeddings, valid_indices = self.extract_embeddings_batch(chunk)

            if embeddings is None:
                print(f"Chunk {chunk_idx} had no valid texts, skipping...")
                continue

            # Save embeddings
            embedding_file = TEXT_EMBEDDINGS_DIR / f"text_embeddings_chunk_{chunk_idx:04d}.npy"
            np.save(embedding_file, embeddings)
            print(f"  Saved embeddings: {embedding_file} ({embeddings.shape})")

            # Save metadata for this chunk
            chunk_metadata = {
                'chunk_idx': chunk_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'valid_indices': valid_indices,
                'valid_texts': [chunk[i] for i in valid_indices],
                'embedding_file': str(embedding_file),
                'num_embeddings': len(embeddings),
                'embedding_dim': embeddings.shape[1],
                'model_used': TEXT_MODEL_NAME,
                'timestamp': datetime.now().isoformat()
            }

            metadata_file = TEXT_METADATA_DIR / f"text_metadata_chunk_{chunk_idx:04d}.json"
            with open(metadata_file, 'w') as f:
                json.dump(chunk_metadata, f, indent=2)
            print(f"  Saved metadata: {metadata_file}")

            # Print chunk statistics
            chunk_confirmed = sum(1 for i in valid_indices if chunk[i]['dataset_type'] == 'confirmed')
            chunk_unconfirmed = len(valid_indices) - chunk_confirmed
            print(f"  Chunk stats: {chunk_confirmed} confirmed, {chunk_unconfirmed} unconfirmed")

        # Save global metadata
        self.save_global_metadata()

    def save_global_metadata(self):
        """Save overall statistics and metadata"""
        global_metadata = {
            'total_texts_processed': self.metadata['dataset_stats']['total'],
            'dataset_stats': self.metadata['dataset_stats'],
            'num_classes': len(self.metadata['class_counts']),
            'class_distribution': dict(self.metadata['class_counts']),
            'top_10_classes': dict(sorted(self.metadata['class_counts'].items(),
                                          key=lambda x: x[1], reverse=True)[:10]),
            'model_used': TEXT_MODEL_NAME,
            'embedding_dim': TEXT_EMBEDDING_DIM,
            'chunk_size': TEXT_CHUNK_SIZE,
            'batch_size': TEXT_BATCH_SIZE,
            'timestamp': datetime.now().isoformat()
        }

        with open(TEXT_METADATA_DIR / 'global_text_metadata.json', 'w') as f:
            json.dump(global_metadata, f, indent=2)

        # Save class counts for popularity weighting
        with open(TEXT_METADATA_DIR / 'text_class_popularity.pkl', 'wb') as f:
            pickle.dump(dict(self.metadata['class_counts']), f)

        print("\n" + "=" * 60)
        print("TEXT EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Total texts processed: {self.metadata['dataset_stats']['total']:,}")
        print(f"Confirmed: {self.metadata['dataset_stats']['confirmed']:,}")
        print(f"Unconfirmed: {self.metadata['dataset_stats']['unconfirmed']:,}")
        print(f"Empty OCR texts skipped: {self.metadata['dataset_stats']['empty_ocr']:,}")
        print(f"Number of unique classes: {len(self.metadata['class_counts']):,}")

        if self.metadata['class_counts']:
            most_popular = max(self.metadata['class_counts'].items(), key=lambda x: x[1])
            print(f"Most popular class: {most_popular[0]} ({most_popular[1]:,} texts)")

        print(f"\nOutputs saved to:")
        print(f"  Embeddings: {TEXT_EMBEDDINGS_DIR}")
        print(f"  Metadata: {TEXT_METADATA_DIR}")
        print("=" * 60)


def main():
    """Main entry point for text extraction"""

    print("\n" + "=" * 80)
    print("MEME TEXT EMBEDDINGS")
    print("=" * 80)

    if not OCR_FILES:
        print("\nNo OCR files configured update OCR_FILES in config.py")
        print("Example:")
        print('  OCR_FILES = ["path/to/confirmed_ocr.csv"]')
        return

    print(f"\nConfigured OCR files:")
    for file in OCR_FILES:
        exists = "Y" if Path(file).exists() else "N"
        print(f"  [{exists}] {file}")

    extractor = MemeTextExtractor()

    # Step 1: Read OCR data from CSV files
    all_texts = extractor.scan_ocr_csv_files()

    if not all_texts:
        print("\nNo valid OCR texts found in the CSV files")
        return

    # Step 2: Process in chunks
    extractor.process_in_chunks(all_texts)

    print("\nText embedding extraction complete")

if __name__ == "__main__":
    main()