"""
Build FAISS indices for text embeddings
Adapted from build_index.py to work with text data
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import numpy as np
import torch
import faiss
import json
import pickle
from tqdm import tqdm
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config.config import RETRIEVAL_CONFIG, Config

# Base paths from config
TEXT_EMBEDDINGS_DIR = RETRIEVAL_CONFIG.TEXT_EMBEDDINGS_DIR
TEXT_METADATA_DIR = RETRIEVAL_CONFIG.TEXT_METADATA_DIR
TEXT_INDEX_DIR = RETRIEVAL_CONFIG.TEXT_INDEX_DIR

# FAISS settings
NLIST = RETRIEVAL_CONFIG.FAISS['nlist']
M = RETRIEVAL_CONFIG.FAISS['m']
BITS = RETRIEVAL_CONFIG.FAISS['bits']

class TextFAISSIndexBuilder:
    """Build FAISS indices for text embeddings"""

    def __init__(self, use_projection=False, checkpoint_path=None):
        """
        Initialize the text index builder

        Args:
            use_projection: If True, apply fine-tuned projection to embeddings
            checkpoint_path: Path to fine-tuned model (if use_projection=True)
        """
        self.use_projection = use_projection
        self.embeddings = []
        self.metadata = []

        # Mappings for text data
        self.idx_to_text = {}
        self.idx_to_text_id = {}
        self.idx_to_class = {}
        self.idx_to_popularity = {}
        self.idx_to_dataset_type = {}
        self.idx_to_split = {}
        self.idx_to_original_path = {}
        self.idx_to_source_file = {}
        self.idx_to_meta_file = {}
        self.idx_to_filename = {}

        # Load projection model if needed (for future fine-tuning)
        if use_projection:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.load_text_projection_model(checkpoint_path)
        else:
            self.projection_model = None
            print("Using RAW text embeddings (baseline)")

    def load_text_projection_model(self, checkpoint_path=None):
        """Load the fine-tuned text projection head"""

        # Import from your actual file name
        from config.models import TextProjectionHead

        if checkpoint_path is None:
            # Use the path from config - points to outputs/retrieval/finetuned_models/text/text_model.pth
            model_path = RETRIEVAL_CONFIG.TEXT_BEST_MODEL

            if not model_path.exists():
                raise FileNotFoundError(
                    f"No fine-tuned text model found at: {model_path}\n"
                    f"Please run finetune_text.py first to create the model.\n"
                    f"Or use use_projection=False for raw embeddings."
                )

            checkpoint_path = model_path
            print(f"Found fine-tuned text model: {checkpoint_path}")
        else:
            # Convert string path to Path object if needed
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Specified checkpoint not found: {checkpoint_path}")
            print(f"Using specified text model: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Initialize model with same config as training
        self.projection_model = TextProjectionHead(
            input_dim=768,
            hidden_dim=checkpoint['config'].get('hidden_dim', 3072),  # Match your training config
            output_dim=checkpoint['config'].get('output_dim', 768),
            num_hidden_layers=2, #HARDCODED TEMPORARY OVERRIDE
            dropout=0.0  # No dropout for inference
        )

        # Load weights
        self.projection_model.load_state_dict(checkpoint['model_state_dict'])
        self.projection_model.to(self.device)
        self.projection_model.eval()

        print(f"Using FINE-TUNED text projection (validation accuracy: {checkpoint.get('val_accuracy', 'N/A'):.4f})")

    def apply_projection_batch(self, embeddings, batch_size=256):
        """Apply projection head to a batch of embeddings"""

        if self.projection_model is None:
            return embeddings  # Return unchanged if no projection

        projected = []

        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                batch_tensor = torch.from_numpy(batch).float().to(self.device)

                # Apply projection and normalize
                proj = self.projection_model(batch_tensor)
                proj = torch.nn.functional.normalize(proj, p=2, dim=1)

                projected.append(proj.cpu().numpy())

        return np.vstack(projected) if projected else embeddings

    def load_all_text_embeddings(self):
        """Load all text embedding chunks and metadata"""
        print("Loading text embeddings and metadata...")

        embedding_files = sorted(TEXT_EMBEDDINGS_DIR.glob("text_embeddings_chunk_*.npy"))
        metadata_files = sorted(TEXT_METADATA_DIR.glob("text_metadata_chunk_*.json"))

        if not embedding_files:
            raise FileNotFoundError(
                f"No text embeddings found in {TEXT_EMBEDDINGS_DIR}! "
                "Please run extract_text_embeddings.py first."
            )

        current_idx = 0
        all_embeddings_raw = []

        # Track statistics
        stats = {
            'total': 0,
            'confirmed': 0,
            'unconfirmed': 0,
            'projected': 0,
            'kept_raw': 0
        }

        for emb_file, meta_file in tqdm(zip(embedding_files, metadata_files),
                                        total=len(embedding_files)):
            # Load raw embeddings
            chunk_embeddings = np.load(emb_file)

            # Load metadata
            with open(meta_file, 'r') as f:
                chunk_metadata = json.load(f)

            # Process each embedding
            processed_embeddings = []
            for i, text_meta in enumerate(chunk_metadata['valid_texts']):
                embedding = chunk_embeddings[i:i + 1]  # Keep as 2D array

                # Check dataset type
                dataset_type = text_meta.get('dataset_type', 'unknown')
                is_confirmed = (dataset_type == 'confirmed')

                # Apply projection ONLY to confirmed texts if fine-tuning enabled
                if self.use_projection and is_confirmed:
                    embedding = self.apply_projection_batch(embedding)
                    stats['projected'] += 1
                elif self.use_projection and not is_confirmed:
                    stats['kept_raw'] += 1

                processed_embeddings.append(embedding)

                # Track statistics
                if is_confirmed:
                    stats['confirmed'] += 1
                else:
                    stats['unconfirmed'] += 1
                stats['total'] += 1

            # Stack processed embeddings for this chunk
            chunk_embeddings_processed = np.vstack(processed_embeddings)
            all_embeddings_raw.append(chunk_embeddings_processed)

            # Build mappings
            for i, text_meta in enumerate(chunk_metadata['valid_texts']):
                idx = current_idx + i

                # Store text content (truncated for display)
                text_content = text_meta.get('text', '')
                self.idx_to_text[idx] = text_content[:500]  # Store first 500 chars

                # Store other metadata
                self.idx_to_text_id[idx] = text_meta.get('text_id', f'text_{idx}')
                self.idx_to_class[idx] = text_meta.get('class_id', 'unknown')
                self.idx_to_popularity[idx] = text_meta.get('popularity', 0)
                self.idx_to_dataset_type[idx] = text_meta.get('dataset_type', 'unknown')
                self.idx_to_original_path[idx] = text_meta.get('original_path', '')
                self.idx_to_source_file[idx] = text_meta.get('source_file', '')
                self.idx_to_split[idx] = None  # Will be assigned later
                self.idx_to_meta_file[idx] = text_meta.get('meta_file', '')
                filename = text_meta.get('original_path', '') or text_meta.get('meta_file', '') or text_meta.get('filename', '')
                self.idx_to_filename[idx] = filename

            current_idx += len(chunk_embeddings_processed)

        # Concatenate all embeddings
        self.embeddings = np.vstack(all_embeddings_raw)

        # Print statistics
        if self.use_projection:
            print(f"\nLoaded {stats['total']} text embeddings with SELECTIVE projection:")
            print(f"  Confirmed texts: {stats['confirmed']} (projection applied)")
            print(f"  Unconfirmed texts: {stats['unconfirmed']} (kept raw)")
        else:
            print(f"\nLoaded {stats['total']} RAW text embeddings:")
            print(f"  Confirmed: {stats['confirmed']}")
            print(f"  Unconfirmed: {stats['unconfirmed']}")

        # Assign train/val splits
        self._assign_splits()

    def _assign_splits(self):
        """Assign 80/20 train/val split to both confirmed and unconfirmed data by class"""
        print("\nAssigning train/val splits to text data...")

        # Process both confirmed and unconfirmed
        for dataset_type in ['confirmed', 'unconfirmed']:
            # Group by class
            by_class = defaultdict(list)
            for idx in range(len(self.embeddings)):
                if self.idx_to_dataset_type[idx] == dataset_type:
                    class_id = self.idx_to_class[idx]
                    by_class[class_id].append(idx)

            # Split each class 80/20
            for class_id, indices in by_class.items():
                n_samples = len(indices)

                if n_samples == 1:
                    self.idx_to_split[indices[0]] = 'train'
                else:
                    n_train = max(1, int(0.8 * n_samples))

                    # Shuffle deterministically
                    np.random.seed(42 + hash(class_id) % 1000)
                    np.random.shuffle(indices)

                    # Assign splits
                    for i, idx in enumerate(indices):
                        if i < n_train:
                            self.idx_to_split[idx] = 'train'
                        else:
                            self.idx_to_split[idx] = 'val'

            print(f"  Assigned splits to {len(by_class)} {dataset_type} classes")

    def get_indices_by_criteria(self, dataset_type=None, split=None):
        """Get indices matching specific criteria"""
        indices = []

        for idx in range(len(self.embeddings)):
            # Check dataset type
            if dataset_type and self.idx_to_dataset_type[idx] != dataset_type:
                continue

            # Check split
            if split and self.idx_to_split[idx] != split:
                continue

            indices.append(idx)

        return indices

    def build_index(self, indices, index_type='IVF_PQ'):
        """Build FAISS index for given indices"""
        # Extract embeddings for these indices
        index_embeddings = self.embeddings[indices]

        d = index_embeddings.shape[1]
        n = len(index_embeddings)

        print(f"  Building {index_type} index for {n} text embeddings...")

        if index_type == 'Flat':
            index = faiss.IndexFlatL2(d)
            index.add(index_embeddings)

        elif index_type == 'IVF_PQ':
            # Adjust nlist if we have fewer embeddings
            nlist = min(NLIST, max(1, int(np.sqrt(n))))

            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFPQ(quantizer, d, nlist, M, BITS)

            print(f"    Training index with {nlist} clusters...")
            index.train(index_embeddings)

            print("    Adding embeddings to index...")
            index.add(index_embeddings)

        elif index_type == 'IVF_Flat':
            # Adjust nlist if we have fewer embeddings
            nlist = min(NLIST, max(1, int(np.sqrt(n))))

            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)

            print(f"    Training index with {nlist} clusters...")
            index.train(index_embeddings)

            print("    Adding embeddings to index...")
            index.add(index_embeddings)

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        return index

    def save_index_with_mappings(self, indices, index_name, index_type='IVF_PQ'):
        """Save FAISS index and comprehensive mappings"""

        if not indices:
            print(f"  No indices for {index_name}, skipping...")
            return None

        # Build index
        index = self.build_index(indices, index_type)

        # Create comprehensive mappings
        mappings = {
            'idx_to_text': {},
            'idx_to_text_id': {},
            'idx_to_class': {},
            'idx_to_filename': {},
            'idx_to_popularity': {},
            'idx_to_dataset_type': {},
            'idx_to_split': {},
            'idx_to_original_path': {},
            'idx_to_source_file': {},
            'idx_to_meta_file': {},
            'original_indices': indices,
            'total_embeddings': len(indices),
            'index_name': index_name,
            'uses_projection': self.use_projection,
            'projection_type': 'fine-tuned' if self.use_projection else 'raw'
        }

        # Map from new index position to original data
        for new_idx, orig_idx in enumerate(indices):
            mappings['idx_to_text'][new_idx] = self.idx_to_text[orig_idx]
            mappings['idx_to_text_id'][new_idx] = self.idx_to_text_id[orig_idx]
            mappings['idx_to_class'][new_idx] = self.idx_to_class[orig_idx]
            mappings['idx_to_popularity'][new_idx] = self.idx_to_popularity[orig_idx]
            mappings['idx_to_dataset_type'][new_idx] = self.idx_to_dataset_type[orig_idx]
            mappings['idx_to_split'][new_idx] = self.idx_to_split[orig_idx]
            mappings['idx_to_original_path'][new_idx] = self.idx_to_original_path[orig_idx]
            mappings['idx_to_source_file'][new_idx] = self.idx_to_source_file[orig_idx]
            mappings['idx_to_meta_file'][new_idx] = self.idx_to_meta_file[orig_idx]
            mappings['idx_to_filename'][new_idx] = self.idx_to_filename[orig_idx]

        # Determine output directory based on projection usage
        if self.use_projection:
            output_dir = TEXT_INDEX_DIR / "finetuned"
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = TEXT_INDEX_DIR

        # Save index
        index_file = output_dir / f"{index_name}_{index_type.lower()}.faiss"
        faiss.write_index(index, str(index_file))
        print(f"    Saved {index_name} index to {index_file}")

        # Save mappings
        mappings_file = output_dir / f"{index_name}_{index_type.lower()}_mappings.pkl"
        with open(mappings_file, 'wb') as f:
            pickle.dump(mappings, f)
        print(f"    Saved mappings to {mappings_file}")

        # Calculate statistics
        stats = self._calculate_statistics(indices)
        stats['index_name'] = index_name
        stats['index_type'] = index_type
        stats['index_size_mb'] = index_file.stat().st_size / 1024 / 1024
        stats['uses_projection'] = self.use_projection

        # Save summary
        summary_file = output_dir / f"{index_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"    Stats: {stats['total_embeddings']} texts, {stats['num_classes']} classes")

        return index

    def _calculate_statistics(self, indices):
        """Calculate detailed statistics for a set of indices"""
        stats = {
            'total_embeddings': len(indices),
            'num_classes': len(set(self.idx_to_class[idx] for idx in indices)),
            'by_dataset_type': {},
            'by_split': {},
            'by_dataset_and_split': {}
        }

        # Count by dataset type
        for idx in indices:
            dt = self.idx_to_dataset_type[idx]
            split = self.idx_to_split[idx]

            # Dataset type counts
            if dt not in stats['by_dataset_type']:
                stats['by_dataset_type'][dt] = 0
            stats['by_dataset_type'][dt] += 1

            # Split counts
            if split not in stats['by_split']:
                stats['by_split'][split] = 0
            stats['by_split'][split] += 1

            # Combined counts
            key = f"{dt}_{split}"
            if key not in stats['by_dataset_and_split']:
                stats['by_dataset_and_split'][key] = 0
            stats['by_dataset_and_split'][key] += 1

        return stats

    def build_all_text_index_combinations(self):
        """Build all text index combinations"""

        if self.use_projection:
            projection_type = "FINE-TUNED text embeddings"
        else:
            projection_type = "RAW text embeddings"

        print("\n" + "=" * 60)
        print(f"Building all TEXT index combinations")
        print(f"Embedding type: {projection_type}")
        print("=" * 60)

        # Define all index combinations we want
        index_configs = [
            # Individual indices
            ('confirmed_text_train', 'confirmed', 'train'),
            ('confirmed_text_val', 'confirmed', 'val'),
            ('unconfirmed_text_train', 'unconfirmed', 'train'),
            ('unconfirmed_text_val', 'unconfirmed', 'val'),

            # Combined indices by dataset type
            ('confirmed_text_all', 'confirmed', None),
            ('unconfirmed_text_all', 'unconfirmed', None),

            # Combined indices by split
            ('text_train_all', None, 'train'),
            ('text_val_all', None, 'val'),

            # Full text index
            ('text_full', None, None)
        ]

        # Build each index
        for index_name, dataset_type, split in index_configs:
            print(f"\n--- Building {index_name} index ---")

            # Add note about projection
            if self.use_projection:
                if dataset_type == 'confirmed':
                    print("  Using FINE-TUNED text projections")
                elif dataset_type == 'unconfirmed':
                    print("  Using RAW embeddings")
                else:
                    print("  Mixed: fine-tuned for confirmed, raw for unconfirmed")

            # Get indices matching criteria
            indices = self.get_indices_by_criteria(dataset_type, split)

            if not indices:
                print(f"  No data found for {index_name}, skipping...")
                continue

            # Build and save
            self.save_index_with_mappings(indices, index_name, 'IVF_PQ')

        print("\n" + "=" * 60)
        print(f"All TEXT indices built successfully!")
        if self.use_projection:
            print("NOTE: Using fine-tuned text projections")
        else:
            print("NOTE: Using raw sentence-transformer embeddings (baseline)")
        print("=" * 60)

        # Print summary of all indices
        self._print_index_summary()

    def _print_index_summary(self):
        """Print summary of all created indices"""
        projection_type = "FINE-TUNED" if self.use_projection else "RAW"
        print(f"\n=== TEXT INDEX SUMMARY ({projection_type}) ===")

        # Determine which directory to check
        if self.use_projection:
            summary_dir = TEXT_INDEX_DIR / "finetuned"
        else:
            summary_dir = TEXT_INDEX_DIR

        if not summary_dir.exists():
            print("No index directory found")
            return

        summary_files = sorted(summary_dir.glob("*_summary.json"))

        if not summary_files:
            print("No summary files found")
            return

        # Load all summaries
        summaries = []
        for f in summary_files:
            with open(f, 'r') as file:
                data = json.load(file)
                data['file'] = f.stem
                summaries.append(data)

        # Sort by total embeddings
        summaries.sort(key=lambda x: x['total_embeddings'], reverse=True)

        # Print table
        print(f"\n{'Index Name':<25} {'Total':<10} {'Classes':<10} {'Size (MB)':<10} {'Breakdown'}")
        print("-" * 85)

        for s in summaries:
            name = s['index_name']
            total = s['total_embeddings']
            classes = s['num_classes']
            size = s.get('index_size_mb', 0)

            # Create breakdown string
            breakdown_parts = []
            for key, count in s.get('by_dataset_and_split', {}).items():
                breakdown_parts.append(f"{key}: {count:,}")
            breakdown = ", ".join(breakdown_parts[:2])  # Show first 2

            print(f"{name:<25} {total:<10,} {classes:<10,} {size:<10.1f} {breakdown}")

        if self.use_projection:
            print("\nNOTE: These indices use FINE-TUNED text projections")
        else:
            print("\nNOTE: These indices use RAW sentence-transformer embeddings (baseline)")


def build_all_text_indices(use_projection=False, checkpoint_path=None):
    """
    Build all text index combinations - can be called from other scripts

    Args:
        use_projection: If True, use fine-tuned projection
                       If False, use raw sentence-transformer embeddings
        checkpoint_path: Optional path to specific checkpoint
    """
    builder = TextFAISSIndexBuilder(use_projection=use_projection, checkpoint_path=checkpoint_path)

    # Load all text embeddings (with projection applied if enabled)
    print("\n[FAISS] Loading text embeddings...")
    builder.load_all_text_embeddings()

    # Build all index combinations
    print("\n[FAISS] Building all text index combinations...")
    builder.build_all_text_index_combinations()

    return builder


def main():
    """Standalone script entry point"""

    # ========== CONFIGURATION ==========
    # Set these flags to control what gets built

    # For initial baseline - build raw indices only
    build_raw = False
    build_finetuned = True  # Set to True after running finetune_text_encoder.py

    # Optional: Specify a particular checkpoint path (usually leave as None)
    checkpoint_path = None

    # Example: To use a specific model file from your retrained data, uncomment and modify:
    # checkpoint_path = "/path/to/your/retrained_text_model/best_model.pth"
    # checkpoint_path = "finetuning/models/new_text_training/final_model.pth"

    # ====================================

    print("\n" + "=" * 80)
    print("TEXT INDEX BUILDER")
    print("=" * 80)

    if build_raw and build_finetuned:
        # Build both raw and fine-tuned indices
        print("\nBuilding COMPLETE TEXT INDEX SET (RAW + FINE-TUNED)")

        # First build raw indices
        print("\n[1/2] Building RAW text embedding indices...")
        build_all_text_indices(use_projection=False)

        # Then build fine-tuned indices
        print("\n[2/2] Building FINE-TUNED text projection indices...")
        try:
            build_all_text_indices(use_projection=True, checkpoint_path=checkpoint_path)
        except FileNotFoundError as e:
            print(f"\nWarning: {e}")
            print("Skipping fine-tuned indices. Run finetune_text.py first.")

        print("\n" + "=" * 80)
        print("ALL TEXT INDICES BUILT SUCCESSFULLY")
        print("Raw indices in: data/text_indices/")
        print("Fine-tuned indices in: data/text_indices/finetuned/")
        print("=" * 80)

    elif build_finetuned:
        # Build only fine-tuned indices
        print("\nBUILDING FINE-TUNED TEXT INDICES")
        try:
            build_all_text_indices(use_projection=True, checkpoint_path=checkpoint_path)
            print("\nFine-tuned text indices saved in: data/text_indices/finetuned/")
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("Please run finetune_text_encoder.py first to train the text projection model.")

    elif build_raw:
        # Build only raw indices
        print("\nBUILDING RAW TEXT INDICES (baseline)")
        build_all_text_indices(use_projection=False)
        print("\nRaw text indices saved in: data/text_indices/")

    else:
        print("No indices to build. Set build_raw=True or build_finetuned=True in main()")


if __name__ == "__main__":
    main()