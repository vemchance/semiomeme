import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import torchvision
torchvision.disable_beta_transforms_warning()

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import pickle
from collections import defaultdict
from transformers import AutoModel, AutoProcessor
from torch.utils.data import Dataset, DataLoader
import hashlib
from datetime import datetime


# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config.config import RETRIEVAL_CONFIG, CORPUS_CONFIG

# Use retrieval config values
MODEL_NAME = RETRIEVAL_CONFIG.IMAGE_MODEL_NAME
EMBEDDING_DIM = RETRIEVAL_CONFIG.IMAGE_EMBEDDING_DIM
BATCH_SIZE = RETRIEVAL_CONFIG.IMAGE_BATCH_SIZE
CHUNK_SIZE = RETRIEVAL_CONFIG.IMAGE_CHUNK_SIZE

CONFIRMED_DIRS = RETRIEVAL_CONFIG.CONFIRMED_DIRS
UNCONFIRMED_DIRS = RETRIEVAL_CONFIG.UNCONFIRMED_DIRS

EMBEDDINGS_DIR = RETRIEVAL_CONFIG.EMBEDDINGS_DIR
METADATA_DIR = RETRIEVAL_CONFIG.METADATA_DIR

SUPPORTED_EXTENSIONS = CORPUS_CONFIG.SUPPORTED_IMAGE_EXTENSIONS

class MemeDataset(Dataset):
    """Simple dataset that handles all memes"""

    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            # Process image using SigLIP processor
            inputs = self.processor(images=image, return_tensors="pt")
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'path': str(path),
                'valid': True
            }
        except Exception as e:
            # Return a dummy tensor for corrupted images
            return {
                'pixel_values': torch.zeros(3, 384, 384),
                'path': str(path),
                'valid': False
            }


class MemeEmbeddingExtractor:
    def __init__(self):
        print("Loading SigLIP model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model.eval()

        # Metadata tracking
        self.metadata = {
            'images': [],  # List of all image metadata
            'class_counts': defaultdict(int),  # Popularity by class
            'dataset_stats': {
                'confirmed': 0,
                'unconfirmed': 0,
                'corrupted': 0,
                'total': 0
            }
        }

    def scan_all_images(self):
        """Scan all image files"""
        print("Scanning all image directories...")
        all_images = []

        SUPPORTED_EXTENSIONS = CORPUS_CONFIG.SUPPORTED_IMAGE_EXTENSIONS

        for dir_path in CONFIRMED_DIRS:
            print(f"Scanning confirmed: {dir_path}")
            for image_path in Path(dir_path).rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    all_images.append({
                        'path': str(image_path),
                        'filename': image_path.name,
                        'class_id': image_path.parent.name,
                        'dataset_type': 'confirmed',  # Still marked as confirmed
                        'dataset_source': str(dir_path)
                    })

        # Scan unconfirmed images
        for dir_path in UNCONFIRMED_DIRS:
            print(f"Scanning unconfirmed: {dir_path}")
            for image_path in Path(dir_path).rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    all_images.append({
                        'path': str(image_path),
                        'filename': image_path.name,
                        'class_id': image_path.parent.name,
                        'dataset_type': 'unconfirmed',
                        'dataset_source': str(dir_path)
                    })

        print(f"Found {len(all_images)} total images")

        # Calculate popularity (class counts)
        for img in all_images:
            self.metadata['class_counts'][img['class_id']] += 1
            self.metadata['dataset_stats'][img['dataset_type']] += 1

        self.metadata['dataset_stats']['total'] = len(all_images)

        # Add popularity score to each image
        for img in all_images:
            img['popularity'] = self.metadata['class_counts'][img['class_id']]
            img['popularity_rank'] = None  # Will calculate after sorting

        # Calculate popularity ranks
        sorted_classes = sorted(self.metadata['class_counts'].items(),
                                key=lambda x: x[1], reverse=True)
        class_to_rank = {cls: rank for rank, (cls, _) in enumerate(sorted_classes)}

        for img in all_images:
            img['popularity_rank'] = class_to_rank[img['class_id']]

        return all_images

    def extract_embeddings_batch(self, image_batch):
        """Extract embeddings for a batch of images"""
        paths = [img['path'] for img in image_batch]
        dataset = MemeDataset(paths, self.processor)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                num_workers=4, pin_memory=True)

        embeddings = []
        valid_indices = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
                pixel_values = batch['pixel_values'].to(self.device)
                valid_mask = batch['valid']

                # Skip entirely invalid batches
                if not valid_mask.any():
                    continue

                # Process valid images
                outputs = self.model.vision_model(pixel_values)
                pooled_output = outputs.pooler_output  # Already normalized by SigLIP

                # Store embeddings and track valid indices
                for idx, (emb, valid) in enumerate(zip(pooled_output, valid_mask)):
                    global_idx = batch_idx * BATCH_SIZE + idx
                    if valid and global_idx < len(image_batch):
                        embeddings.append(emb.cpu().numpy())
                        valid_indices.append(global_idx)
                    elif not valid and global_idx < len(image_batch):
                        self.metadata['dataset_stats']['corrupted'] += 1

        return np.vstack(embeddings) if embeddings else None, valid_indices

    def process_in_chunks(self, all_images):
        """Process images in chunks to manage memory"""
        Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)
        Path(METADATA_DIR).mkdir(parents=True, exist_ok=True)

        num_chunks = (len(all_images) + CHUNK_SIZE - 1) // CHUNK_SIZE

        print(f"Processing {len(all_images)} images in {num_chunks} chunks...")

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * CHUNK_SIZE
            end_idx = min(start_idx + CHUNK_SIZE, len(all_images))
            chunk = all_images[start_idx:end_idx]

            print(f"\nChunk {chunk_idx + 1}/{num_chunks}: Processing images {start_idx}-{end_idx}")

            # Extract embeddings
            embeddings, valid_indices = self.extract_embeddings_batch(chunk)

            if embeddings is None:
                print(f"Chunk {chunk_idx} had no valid images, skipping...")
                continue

            # Save embeddings
            embedding_file = EMBEDDINGS_DIR / f"embeddings_chunk_{chunk_idx:04d}.npy"
            np.save(embedding_file, embeddings)

            # Save metadata for this chunk
            chunk_metadata = {
                'chunk_idx': chunk_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'valid_indices': valid_indices,
                'valid_images': [chunk[i] for i in valid_indices],
                'embedding_file': str(embedding_file),
                'num_embeddings': len(embeddings),
                'timestamp': datetime.now().isoformat()
            }

            metadata_file = METADATA_DIR / f"metadata_chunk_{chunk_idx:04d}.json"
            with open(metadata_file, 'w') as f:
                json.dump(chunk_metadata, f, indent=2)

            print(f"Saved {len(embeddings)} embeddings and metadata for chunk {chunk_idx}")

        # Save global metadata
        self.save_global_metadata()

    def save_global_metadata(self):
        """Save overall statistics and metadata"""
        global_metadata = {
            'total_images_scanned': self.metadata['dataset_stats']['total'],
            'dataset_stats': self.metadata['dataset_stats'],
            'num_classes': len(self.metadata['class_counts']),
            'class_distribution': dict(self.metadata['class_counts']),
            'top_10_classes': dict(sorted(self.metadata['class_counts'].items(),
                                          key=lambda x: x[1], reverse=True)[:10]),
            'model_used': MODEL_NAME,
            'embedding_dim': EMBEDDING_DIM,
            'chunk_size': CHUNK_SIZE,
            'timestamp': datetime.now().isoformat()
        }

        with open(METADATA_DIR / 'global_metadata.json', 'w') as f:
            json.dump(global_metadata, f, indent=2)

        # Save class counts for popularity weighting
        with open(METADATA_DIR / 'class_popularity.pkl', 'wb') as f:
            pickle.dump(dict(self.metadata['class_counts']), f)

        print("\n=== Extraction Complete ===")
        print(f"Total images scanned: {self.metadata['dataset_stats']['total']}")
        print(f"Confirmed: {self.metadata['dataset_stats']['confirmed']}")
        print(f"Unconfirmed: {self.metadata['dataset_stats']['unconfirmed']}")
        print(f"Corrupted/skipped: {self.metadata['dataset_stats']['corrupted']}")
        print(f"Number of unique classes: {len(self.metadata['class_counts'])}")
        print(f"Most popular class: {max(self.metadata['class_counts'].items(), key=lambda x: x[1])}")


def main():
    extractor = MemeEmbeddingExtractor()

    # Step 1: Scan all images
    all_images = extractor.scan_all_images()

    # Step 2: Process in chunks
    extractor.process_in_chunks(all_images)

    print("\nEmbedding extraction complete!")
    print(f"Embeddings saved to: {EMBEDDINGS_DIR}")
    print(f"Metadata saved to: {METADATA_DIR}")


if __name__ == "__main__":
    main()