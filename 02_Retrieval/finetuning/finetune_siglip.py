"""
Fine-tuning using ONLY confirmed memes from cached embeddings
Improved version with better model naming and no checkpoint saving
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import json
import random
import sys

# PyTorch Metric Learning
from pytorch_metric_learning import distances, losses, miners
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

sys.path.append(str(Path(__file__).parent.parent))
from config.config import RETRIEVAL_CONFIG
EMBEDDINGS_DIR = RETRIEVAL_CONFIG.EMBEDDINGS_DIR
METADATA_DIR = RETRIEVAL_CONFIG.METADATA_DIR
INDEX_DIR = RETRIEVAL_CONFIG.INDEX_DIR

# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_CONFIG = {
    # Data parameters
    'batch_size': 8192,
    'val_batch_size': 2048,
    'num_epochs': 200,
    'random_seed': 42,
    'accumulation_steps': 4,

    # FILTERING - THIS IS KEY!
    'use_only_confirmed': True,
    'confirmed_path_pattern': 'Confirmed Images',
    'exclude_pattern': 'Unconfirmed',

    # Model parameters
    'input_dim': 768,
    'hidden_dim': 3072,
    'output_dim': 768,
    'num_hidden_layers': 3,
    'dropout': 0.2,

    # Optimizer parameters
    'learning_rate': 1e-3,
    'weight_decay': 0.01,

    'loss_type': 'SupConLoss',
    'temperature': 0.05,
    'miner_type': None,
    'miner_epsilon': 0.1,  # Not used for TripletMarginMiner
    'miner_margin': 0.7,
    'miner_triplet_type': 'semihard',

    # System parameters
    'num_workers': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Saving parameters - NO CHECKPOINTS
    'save_dir': RETRIEVAL_CONFIG.OUTPUT_DIR / 'finetuned_models' / 'vision',
    'save_checkpoints': False,  # Disabled
    'patience': 10,

    # Validation
    'validate_every_n_epochs': 1,
}


# ============================================================================
# DATASET
# ============================================================================

class ConfirmedOnlyEmbeddingDataset(Dataset):
    """Dataset that loads ONLY confirmed meme embeddings"""

    def __init__(self, split='train', max_chunks=None, verbose=True):
        self.split = split
        self.samples = []
        self.class_to_idx = {}

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Loading {split.upper()} Dataset - CONFIRMED ONLY")
            print('=' * 60)

        # Get all chunk files
        embedding_files = sorted(EMBEDDINGS_DIR.glob("embeddings_chunk_*.npy"))
        metadata_files = sorted(METADATA_DIR.glob("metadata_chunk_*.json"))

        if max_chunks:
            embedding_files = embedding_files[:max_chunks]
            metadata_files = metadata_files[:max_chunks]

        print(f"Processing {len(embedding_files)} chunks...")

        # Temporary storage
        all_embeddings = []
        all_metadata = []

        # Statistics
        total_images = 0
        confirmed_images = 0
        unconfirmed_images = 0

        # Load chunks and filter
        for emb_file, meta_file in tqdm(zip(embedding_files, metadata_files),
                                        total=len(embedding_files),
                                        desc="Loading and filtering"):
            # Load embeddings chunk
            chunk_embeddings = np.load(emb_file)

            # Load metadata
            with open(meta_file, 'r') as f:
                chunk_metadata = json.load(f)

            # FILTER: Only keep confirmed images
            for i, img_meta in enumerate(chunk_metadata['valid_images']):
                total_images += 1
                path = img_meta['path']

                # Check if this is a confirmed image
                is_confirmed = (
                        TRAIN_CONFIG['confirmed_path_pattern'] in path and
                        TRAIN_CONFIG['exclude_pattern'] not in path
                )

                if is_confirmed:
                    confirmed_images += 1
                    all_embeddings.append(chunk_embeddings[i])
                    all_metadata.append(img_meta)
                else:
                    unconfirmed_images += 1

        print(f"\nFiltering Results:")
        print(f"  Total images scanned: {total_images:,}")
        print(f"  Confirmed images kept: {confirmed_images:,}")
        print(f"  Unconfirmed images excluded: {unconfirmed_images:,}")

        if confirmed_images == 0:
            raise ValueError("No confirmed images found.")

        # Stack all confirmed embeddings
        self.embeddings = np.vstack(all_embeddings)
        print(f"\nLoaded embeddings shape: {self.embeddings.shape}")

        # Build class mapping from confirmed only
        unique_classes = set(meta['class_id'] for meta in all_metadata)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        print(f"Unique classes: {len(self.class_to_idx)}")

        # Group samples by class for proper splitting
        class_to_samples = defaultdict(list)
        for idx, meta in enumerate(all_metadata):
            class_id = meta['class_id']
            class_to_samples[class_id].append({
                'embedding_idx': idx,
                'meta': meta
            })

        # Split each class 80/20
        np.random.seed(TRAIN_CONFIG['random_seed'])

        for class_id, class_samples in class_to_samples.items():
            n_samples = len(class_samples)

            if n_samples == 1:
                # Only one sample - goes to train
                if split == 'train':
                    sample_data = class_samples[0]
                    self.samples.append({
                        'embedding_idx': sample_data['embedding_idx'],
                        'label': self.class_to_idx[class_id],
                        'class_id': class_id,
                        'path': sample_data['meta']['path']
                    })
            else:
                # Multiple samples - stratified split
                class_seed = 42 + hash(class_id) % 10000
                rng = np.random.RandomState(class_seed)
                indices = rng.permutation(n_samples)

                n_train = max(1, int(0.8 * n_samples))

                if split == 'train':
                    selected_indices = indices[:n_train]
                else:  # val
                    selected_indices = indices[n_train:]

                for idx in selected_indices:
                    sample_data = class_samples[idx]
                    self.samples.append({
                        'embedding_idx': sample_data['embedding_idx'],
                        'label': self.class_to_idx[class_id],
                        'class_id': class_id,
                        'path': sample_data['meta']['path']
                    })

        print(f"\n{split.upper()} split: {len(self.samples)} samples")

        # Calculate class distribution
        class_counts = defaultdict(int)
        for sample in self.samples:
            class_counts[sample['label']] += 1

        print(f"Class distribution:")
        print(f"  Min samples per class: {min(class_counts.values())}")
        print(f"  Max samples per class: {max(class_counts.values())}")
        print(f"  Avg samples per class: {np.mean(list(class_counts.values())):.1f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        embedding = self.embeddings[sample['embedding_idx']]
        label = sample['label']
        return torch.from_numpy(embedding).float(), label


# ============================================================================
# MODEL
# ============================================================================

class ProjectionHead(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=3072, output_dim=768,
                 num_hidden_layers=3, dropout=0.1):
        super().__init__()

        self.layers = nn.Sequential(
            # First expansion
            nn.Linear(input_dim, hidden_dim),  # 768 -> 3072
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Middle layer (stays wide)
            nn.Linear(hidden_dim, hidden_dim),  # 3072 -> 3072
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Final projection
            nn.Linear(hidden_dim, output_dim),  # 3072 -> 768
            nn.BatchNorm1d(output_dim)
        )

        # Very small or no residual
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        transformed = self.layers(x)
        if self.alpha > 0:
            out = transformed + self.alpha * x
        else:
            out = transformed
        out = F.normalize(out, p=2, dim=1)
        return out


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_descriptor(config):
    """
    Create a descriptive string for the model configuration
    """
    # Shorten names for readability
    loss_map = {
        'MultiSimilarityLoss': 'MSLoss',
        'NTXentLoss': 'NTXent',
        'TripletMarginLoss': 'Triplet'
    }

    miner_map = {
        'MultiSimilarityMiner': 'MSMiner',
        'TripletMarginMiner': 'TripMiner',
        None: 'NoMiner'
    }

    loss_short = loss_map.get(config['loss_type'], config['loss_type'])
    miner_short = miner_map.get(config['miner_type'], 'NoMiner')

    # Create descriptor
    descriptor = f"{loss_short}_{miner_short}_{config['hidden_dim']}d"

    if config['num_hidden_layers'] > 1:
        descriptor += f"x{config['num_hidden_layers']}"

    return descriptor


def get_loss_and_miner(config):
    """Initialize loss function and miner for training only"""

    distance = distances.CosineSimilarity()

    if config['loss_type'] == 'SupConLoss':
        loss_fn = losses.SupConLoss(
            temperature=0.05,  # Lower temperature often better
            distance=distance
        )
    elif config['loss_type'] == 'MultiSimilarityLoss':
        loss_fn = losses.MultiSimilarityLoss(
            alpha=2, beta=50, base=0.5, distance=distance
        )
    elif config['loss_type'] == 'NTXentLoss':
        loss_fn = losses.NTXentLoss(
            temperature=config['temperature'], distance=distance
        )
    elif config['loss_type'] == 'TripletMarginLoss':
        loss_fn = losses.TripletMarginLoss(
            margin=0.1, distance=distance
        )
    else:
        raise ValueError(f"Unknown loss type: {config['loss_type']}")

    # Miner - CHANGE THIS PART
    if config['miner_type'] == 'MultiSimilarityMiner':
        miner = miners.MultiSimilarityMiner(
            epsilon=config['miner_epsilon'], distance=distance
        )
    elif config['miner_type'] == 'TripletMarginMiner':
        miner = miners.TripletMarginMiner(
            margin=config.get('miner_margin', 0.5),  # Use config value, default 0.5
            distance=distance,
            type_of_triplets=config.get('miner_triplet_type', 'hard')  # Use 'hard' instead of 'semihard'
        )
    else:
        miner = None

    print(f"Loss function: {config['loss_type']}")
    print(f"Miner: {config['miner_type'] if miner else 'None'}")
    if config['miner_type'] == 'TripletMarginMiner':
        print(f"  - Margin: {config.get('miner_margin', 0.5)}")
        print(f"  - Triplet type: {config.get('miner_triplet_type', 'hard')}")

    return loss_fn, miner


def validate_retrieval_accuracy_with_faiss(model, train_loader, val_loader, device):
    """
    Validation using FAISS indices
    """
    import faiss

    model.eval()

    # Collect train embeddings and build FAISS index
    train_embeddings = []
    train_labels = []

    with torch.no_grad():
        for embeddings, labels in tqdm(train_loader, desc="  Collecting train embeddings", leave=False):
            embeddings = embeddings.to(device)
            projected = model(embeddings)
            projected = torch.nn.functional.normalize(projected, p=2, dim=1)
            train_embeddings.append(projected.cpu().numpy())
            train_labels.extend(labels.numpy())

    train_embeddings = np.vstack(train_embeddings).astype('float32')
    train_labels = np.array(train_labels)

    # Build FAISS index (Flat for accuracy)
    print(f"  Building FAISS index with {len(train_embeddings)} vectors...", end="")
    d = train_embeddings.shape[1]
    train_index = faiss.IndexFlatL2(d)
    train_index.add(train_embeddings)
    print(" Done")

    # Collect val embeddings
    val_embeddings = []
    val_labels = []

    with torch.no_grad():
        for embeddings, labels in tqdm(val_loader, desc="  Collecting val embeddings", leave=False):
            embeddings = embeddings.to(device)
            projected = model(embeddings)
            projected = torch.nn.functional.normalize(projected, p=2, dim=1)
            val_embeddings.append(projected.cpu().numpy())
            val_labels.extend(labels.numpy())

    val_embeddings = np.vstack(val_embeddings).astype('float32')
    val_labels = np.array(val_labels)

    # Search val queries in train index with k=10 for recall metrics
    print(f"  Searching {len(val_embeddings)} queries in FAISS index...", end="")
    k_max = 10  # Get top-10 for recall calculation
    distances, indices = train_index.search(val_embeddings, k_max)
    print(" Done")

    # Calculate Recall@K metrics
    recalls = {}
    k_values = [1, 5, 10]

    for k in k_values:
        correct = 0
        for i in range(len(val_labels)):
            # Check if true label appears in top-k predictions
            top_k_labels = train_labels[indices[i, :k]]
            if val_labels[i] in top_k_labels:
                correct += 1
        recalls[k] = correct / len(val_labels)

    # Top-1 accuracy (same as Recall@1)
    accuracy = recalls[1]

    # Print metrics
    print(f"    Accuracy (Recall@1): {accuracy:.4f}")
    print(f"    Recall@5: {recalls[5]:.4f}")
    print(f"    Recall@10: {recalls[10]:.4f}")

    model.train()

    # Return dict with all metrics
    return {
        'accuracy': accuracy,
        'recall@1': recalls[1],
        'recall@5': recalls[5],
        'recall@10': recalls[10]
    }


def validate_retrieval_accuracy(model, val_loader, device, train_loader=None):
    """
    FAISS for evaluation
    Falls back to simple similarity if train_loader not provided
    """
    if train_loader is not None:
        return validate_retrieval_accuracy_with_faiss(model, train_loader, val_loader, device)

    # Fallback to original implementation
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for embeddings, labels in tqdm(val_loader, desc="Collecting val embeddings", leave=False):
            embeddings = embeddings.to(device)
            projected = model(embeddings)
            projected = torch.nn.functional.normalize(projected, p=2, dim=1)
            all_embeddings.append(projected)
            all_labels.append(labels.to(device))

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    n_samples = len(all_embeddings)

    # Simple self-similarity (old method)
    if n_samples > 10000:
        correct = 0
        chunk_size = 1000

        for i in tqdm(range(0, n_samples, chunk_size), desc="Computing accuracy", leave=False):
            end_idx = min(i + chunk_size, n_samples)
            chunk_embeddings = all_embeddings[i:end_idx]
            chunk_labels = all_labels[i:end_idx]

            distances = torch.cdist(chunk_embeddings, all_embeddings)

            for j in range(len(chunk_embeddings)):
                distances[j, i + j] = float('inf')

            nearest_indices = distances.argmin(dim=1)
            nearest_labels = all_labels[nearest_indices]

            correct += (nearest_labels == chunk_labels).sum().item()

        accuracy = correct / n_samples
    else:
        distances = torch.cdist(all_embeddings, all_embeddings)
        distances.fill_diagonal_(float('inf'))
        nearest_indices = distances.argmin(dim=1)
        nearest_labels = all_labels[nearest_indices]
        accuracy = (nearest_labels == all_labels).float().mean().item()

    del all_embeddings
    del all_labels
    torch.cuda.empty_cache()

    model.train()
    return accuracy


def load_projection_head(checkpoint_path=None, model_descriptor=None):
    """Load trained projection head for inference"""

    if checkpoint_path is None:
        # Try to find the best model based on descriptor
        if model_descriptor:
            search_pattern = f"*{model_descriptor}*/best_model.pth"
            candidates = list(TRAIN_CONFIG['save_dir'].glob(search_pattern))
            if candidates:
                checkpoint_path = candidates[-1]  # Use most recent
            else:
                raise FileNotFoundError(f"No model found for descriptor: {model_descriptor}")
        else:
            # Fallback to most recent best model
            candidates = list(TRAIN_CONFIG['save_dir'].glob("*/best_model.pth"))
            if candidates:
                checkpoint_path = candidates[-1]
            else:
                raise FileNotFoundError("No trained models found")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    # Initialize model
    model = ProjectionHead(
        input_dim=TRAIN_CONFIG['input_dim'],
        hidden_dim=TRAIN_CONFIG['hidden_dim'],
        output_dim=TRAIN_CONFIG['output_dim'],
        num_hidden_layers=TRAIN_CONFIG['num_hidden_layers'],
        dropout=TRAIN_CONFIG['dropout']
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded model from: {checkpoint_path}")
    if 'model_descriptor' in checkpoint:
        print(f"Model configuration: {checkpoint['model_descriptor']}")
    if 'val_accuracy' in checkpoint:
        print(f"Model validation accuracy: {checkpoint['val_accuracy']:.4f}")
    if 'dataset_info' in checkpoint:
        info = checkpoint['dataset_info']
        print(f"Trained on: {info['num_train']} train, {info['num_val']} val samples")
        print(f"Number of classes: {info['num_classes']}")

    return model


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train():
    """Main training function with improved naming"""

    # Get model descriptor for naming
    model_descriptor = get_model_descriptor(TRAIN_CONFIG)

    print("\n" + "=" * 70)
    print("FINE-TUNING")
    print(f"Configuration: {model_descriptor}")
    print("=" * 70)

    device = torch.device(TRAIN_CONFIG['device'])
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directory with model descriptor
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = TRAIN_CONFIG['save_dir'] / f"{model_descriptor}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSave directory: {save_dir}")

    # Save configuration
    with open(save_dir / 'config.json', 'w') as f:
        config_to_save = {k: str(v) if isinstance(v, Path) else v
                          for k, v in TRAIN_CONFIG.items()}
        config_to_save['model_descriptor'] = model_descriptor
        config_to_save['timestamp'] = timestamp
        json.dump(config_to_save, f, indent=2)

    # Load datasets
    train_dataset = ConfirmedOnlyEmbeddingDataset(split='train')
    val_dataset = ConfirmedOnlyEmbeddingDataset(split='val')

    print(f"\n{'=' * 60}")
    print(f"Dataset Summary:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Total classes: {len(train_dataset.class_to_idx)}")
    print('=' * 60)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG['val_batch_size'],
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers'],
        pin_memory=True
    )

    print(f"\nBatches per epoch: {len(train_loader)} train, {len(val_loader)} val")

    # Initialize model
    print("\nInitialising projection head...")
    model = ProjectionHead(
        input_dim=TRAIN_CONFIG['input_dim'],
        hidden_dim=TRAIN_CONFIG['hidden_dim'],
        output_dim=TRAIN_CONFIG['output_dim'],
        num_hidden_layers=TRAIN_CONFIG['num_hidden_layers'],
        dropout=TRAIN_CONFIG['dropout']
    )
    model = model.to(device)
    model.train()

    # Loss and miner
    loss_fn, miner = get_loss_and_miner(TRAIN_CONFIG)
    loss_fn = loss_fn.to(device)
    if miner:
        miner = miner.to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )

    # Scheduler
    num_training_steps = len(train_loader) * TRAIN_CONFIG['num_epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)

    # Training state
    best_val_accuracy = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_accuracy': [],
        'learning_rates': [],
        'config': model_descriptor
    }

    # Training loop
    print(f"\nStarting training for {TRAIN_CONFIG['num_epochs']} epochs...")
    print("-" * 70)

    for epoch in range(TRAIN_CONFIG['num_epochs']):
        epoch_loss = 0
        num_batches = 0

        # Training
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{TRAIN_CONFIG['num_epochs']}")


        for batch_idx, (embeddings, labels) in enumerate(pbar):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            projected = model(embeddings)

            # Mining and loss
            if miner:
                hard_pairs = miner(projected, labels)
                loss = loss_fn(projected, labels, hard_pairs)
            else:
                loss = loss_fn(projected, labels)

            # Skip if loss is invalid
            if not torch.isfinite(loss) or loss.item() < 1e-8:
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update
            optimizer.step()
            scheduler.step()

            # Track
            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })

        # Epoch metrics
        avg_train_loss = epoch_loss / max(num_batches, 1)
        history['train_loss'].append(avg_train_loss)
        history['learning_rates'].append(scheduler.get_last_lr()[0])

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        # Validation
        if (epoch + 1) % TRAIN_CONFIG['validate_every_n_epochs'] == 0:
            # Pass both train and val loaders for proper evaluation
            val_metrics = validate_retrieval_accuracy(model, val_loader, device, train_loader)
            val_accuracy = val_metrics['accuracy']  # For backward compatibility
            history['val_accuracy'].append(val_accuracy)
            history['val_recall@5'] = history.get('val_recall@5', [])
            history['val_recall@10'] = history.get('val_recall@10', [])
            history['val_recall@5'].append(val_metrics['recall@5'])
            history['val_recall@10'].append(val_metrics['recall@10'])

            print(f"  Val Accuracy: {val_accuracy:.4f}")
            print(f"  Val Recall@5: {val_metrics['recall@5']:.4f}")
            print(f"  Val Recall@10: {val_metrics['recall@10']:.4f}")

            # Save best model with descriptive name
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0

                # Save with descriptive name including accuracy
                best_model_name = f'best_{model_descriptor}_acc{val_accuracy:.4f}.pth'
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_accuracy': val_accuracy,
                    'train_loss': avg_train_loss,
                    'config': TRAIN_CONFIG,
                    'model_descriptor': model_descriptor,
                    'dataset_info': {
                        'num_train': len(train_dataset),
                        'num_val': len(val_dataset),
                        'num_classes': len(train_dataset.class_to_idx),
                        'confirmed_only': True
                    }
                }

                save_path = save_dir / best_model_name
                torch.save(checkpoint, save_path)
                print(f"  [SAVED] New best model: {best_model_name}")

                # Also save as 'best_model.pth' for easy loading
                torch.save(checkpoint, save_dir / 'vision_model.pth')
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{TRAIN_CONFIG['patience']}")

        # Early stopping
        if patience_counter >= TRAIN_CONFIG['patience']:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation accuracy: {best_val_accuracy:.4f}")
            break

    # Save final model with descriptive name
    final_model_name = f'final_{model_descriptor}_epochs{epoch + 1}_acc{best_val_accuracy:.4f}.pth'
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': TRAIN_CONFIG,
        'model_descriptor': model_descriptor,
        'final_epoch': epoch + 1,
        'best_val_accuracy': best_val_accuracy
    }

    torch.save(final_checkpoint, save_dir / final_model_name)

    # Save training history with descriptive name
    history_name = f'history_{model_descriptor}.json'
    with open(save_dir / history_name, 'w') as f:
        json.dump(history, f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print("=" * 70)
    print(f"Configuration: {model_descriptor}")
    print(f"Total epochs: {epoch + 1}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Models saved to: {save_dir}")
    print(f"  Best model: best_{model_descriptor}_acc{best_val_accuracy:.4f}.pth")
    print(f"  Final model: {final_model_name}")
    print(f"  Update models.py with new configuration")
    print("=" * 70)

    return model, history


def main():
    """Entry point"""
    return train()


if __name__ == "__main__":
    main()