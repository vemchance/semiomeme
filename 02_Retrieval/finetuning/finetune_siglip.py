"""
Fine-tuning vision embeddings with configurable data source
Supports: confirmed, unconfirmed, or combined meme datasets
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
# TRAINING MODE - SET THIS BEFORE RUNNING
# ============================================================================
# Options: 'confirmed', 'unconfirmed', 'combined'
TRAINING_MODE = 'combined'
# ============================================================================

# ============================================================================
# CONFIGURATION
# ============================================================================

def get_save_dir(mode):
    """Get save directory based on training mode"""
    return RETRIEVAL_CONFIG.OUTPUT_DIR / 'finetuned_models' / 'vision' / mode


TRAIN_CONFIG = {
    # Data parameters
    'batch_size': 8192,
    'val_batch_size': 2048,
    'num_epochs': 10,
    'random_seed': 42,
    'accumulation_steps': 4,

    # Filtering patterns
    'confirmed_path_pattern': 'Confirmed Images',
    'unconfirmed_path_pattern': 'Unconfirmed Images',
    'exclude_pattern': 'None',

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
    'miner_epsilon': 0.1,
    'miner_margin': 0.7,
    'miner_triplet_type': 'semihard',

    # System parameters
    'num_workers': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Saving parameters
    'save_dir': get_save_dir(TRAINING_MODE),
    'save_checkpoints': False,
    'patience': 10,

    # Validation
    'validate_every_n_epochs': 1,
}


# ============================================================================
# DATASET
# ============================================================================

class EmbeddingDataset(Dataset):
    """Dataset that loads embeddings based on training mode"""

    def __init__(self, split='train', mode='confirmed', max_chunks=None, verbose=True):
        self.split = split
        self.mode = mode
        self.samples = []
        self.class_to_idx = {}

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Loading {split.upper()} Dataset - MODE: {mode.upper()}")
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
        included_images = 0

        # Load chunks and filter
        for emb_file, meta_file in tqdm(zip(embedding_files, metadata_files),
                                        total=len(embedding_files),
                                        desc="Loading and filtering"):
            chunk_embeddings = np.load(emb_file)

            with open(meta_file, 'r') as f:
                chunk_metadata = json.load(f)

            for i, img_meta in enumerate(chunk_metadata['valid_images']):
                total_images += 1
                path = img_meta['path']

                # Determine image type
                is_confirmed = TRAIN_CONFIG['confirmed_path_pattern'] in path
                is_unconfirmed = TRAIN_CONFIG['unconfirmed_path_pattern'] in path

                if is_confirmed:
                    confirmed_images += 1
                elif is_unconfirmed:
                    unconfirmed_images += 1

                # Filter based on mode
                include = False
                if mode == 'confirmed' and is_confirmed:
                    include = True
                elif mode == 'unconfirmed' and is_unconfirmed:
                    include = True
                elif mode == 'combined' and (is_confirmed or is_unconfirmed):
                    include = True

                # Apply exclude pattern
                if include and TRAIN_CONFIG['exclude_pattern'] in path:
                    include = False

                if include:
                    included_images += 1
                    all_embeddings.append(chunk_embeddings[i])
                    all_metadata.append(img_meta)

        print(f"\nFiltering Results:")
        print(f"  Total images scanned: {total_images:,}")
        print(f"  Confirmed images: {confirmed_images:,}")
        print(f"  Unconfirmed images: {unconfirmed_images:,}")
        print(f"  Included for training ({mode}): {included_images:,}")

        if included_images == 0:
            raise ValueError(f"No images found for mode: {mode}")

        # Stack all embeddings
        self.embeddings = np.vstack(all_embeddings)
        print(f"\nLoaded embeddings shape: {self.embeddings.shape}")

        # Build class mapping
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
                if split == 'train':
                    sample_data = class_samples[0]
                    self.samples.append({
                        'embedding_idx': sample_data['embedding_idx'],
                        'label': self.class_to_idx[class_id],
                        'class_id': class_id,
                        'path': sample_data['meta']['path']
                    })
            else:
                class_seed = 42 + hash(class_id) % 10000
                rng = np.random.RandomState(class_seed)
                indices = rng.permutation(n_samples)

                n_train = max(1, int(0.8 * n_samples))

                if split == 'train':
                    selected_indices = indices[:n_train]
                else:
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
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )

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

def get_model_descriptor(config, mode):
    """Create a descriptive string for the model configuration"""
    loss_map = {
        'MultiSimilarityLoss': 'MSLoss',
        'NTXentLoss': 'NTXent',
        'TripletMarginLoss': 'Triplet',
        'SupConLoss': 'SupCon'
    }

    miner_map = {
        'MultiSimilarityMiner': 'MSMiner',
        'TripletMarginMiner': 'TripMiner',
        None: 'NoMiner'
    }

    loss_short = loss_map.get(config['loss_type'], config['loss_type'])
    miner_short = miner_map.get(config['miner_type'], 'NoMiner')

    descriptor = f"{mode}_{loss_short}_{miner_short}_{config['hidden_dim']}d"

    if config['num_hidden_layers'] > 1:
        descriptor += f"x{config['num_hidden_layers']}"

    return descriptor


def get_loss_and_miner(config):
    """Initialize loss function and miner"""
    distance = distances.CosineSimilarity()

    if config['loss_type'] == 'SupConLoss':
        loss_fn = losses.SupConLoss(
            temperature=0.05,
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

    if config['miner_type'] == 'MultiSimilarityMiner':
        miner = miners.MultiSimilarityMiner(
            epsilon=config['miner_epsilon'], distance=distance
        )
    elif config['miner_type'] == 'TripletMarginMiner':
        miner = miners.TripletMarginMiner(
            margin=config.get('miner_margin', 0.5),
            distance=distance,
            type_of_triplets=config.get('miner_triplet_type', 'hard')
        )
    else:
        miner = None

    print(f"Loss function: {config['loss_type']}")
    if miner:
        print(f"Miner: {config['miner_type']}")
    else:
        print("No miner")

    return loss_fn, miner


def validate_retrieval_accuracy(model, val_loader, device, train_loader=None):
    """Validate using retrieval accuracy, recall, and MRR"""
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings = embeddings.to(device)
            projected = model(embeddings)
            all_embeddings.append(projected.cpu())
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute similarities
    similarities = torch.mm(all_embeddings, all_embeddings.t())
    similarities.fill_diagonal_(-float('inf'))

    # Get top-k indices
    _, top1_indices = similarities.topk(1, dim=1)
    _, top5_indices = similarities.topk(5, dim=1)
    _, top10_indices = similarities.topk(10, dim=1)

    # Recall@1 (same as accuracy)
    predictions = all_labels[top1_indices.squeeze()]
    recall_1 = (predictions == all_labels).float().mean().item()

    # Recall@5 and Recall@10
    recall_5 = 0
    recall_10 = 0
    for i in range(len(all_labels)):
        if all_labels[i] in all_labels[top5_indices[i]]:
            recall_5 += 1
        if all_labels[i] in all_labels[top10_indices[i]]:
            recall_10 += 1

    recall_5 /= len(all_labels)
    recall_10 /= len(all_labels)

    # MRR - Mean Reciprocal Rank
    # For each query, find rank of first correct match
    _, all_indices = similarities.topk(similarities.size(1), dim=1)
    reciprocal_ranks = []
    for i in range(len(all_labels)):
        query_label = all_labels[i]
        retrieved_labels = all_labels[all_indices[i]]
        # Find first position where label matches
        matches = (retrieved_labels == query_label).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            rank = matches[0].item() + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    mrr = np.mean(reciprocal_ranks)

    model.train()

    return {
        'accuracy': recall_1,
        'recall@1': recall_1,
        'recall@5': recall_5,
        'recall@10': recall_10,
        'mrr': mrr
    }


# ============================================================================
# TRAINING
# ============================================================================

def train(mode=None):
    """Main training function"""
    if mode is None:
        mode = TRAINING_MODE

    print(f"\n{'=' * 70}")
    print(f"VISION EMBEDDING FINE-TUNING")
    print(f"Mode: {mode.upper()}")
    print('=' * 70)

    device = torch.device(TRAIN_CONFIG['device'])
    print(f"Using device: {device}")

    # Update save directory for this mode
    save_dir = get_save_dir(mode)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create model descriptor
    model_descriptor = get_model_descriptor(TRAIN_CONFIG, mode)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nModel configuration: {model_descriptor}")
    print(f"Saving to: {save_dir}")

    # Save configuration
    with open(save_dir / 'config.json', 'w') as f:
        config_to_save = {k: str(v) if isinstance(v, Path) else v
                          for k, v in TRAIN_CONFIG.items()}
        config_to_save['mode'] = mode
        config_to_save['model_descriptor'] = model_descriptor
        config_to_save['timestamp'] = timestamp
        json.dump(config_to_save, f, indent=2)

    # Load datasets with mode
    train_dataset = EmbeddingDataset(split='train', mode=mode)
    val_dataset = EmbeddingDataset(split='val', mode=mode)

    print(f"\n{'=' * 60}")
    print(f"Dataset Summary:")
    print(f"  Mode: {mode}")
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
        'config': model_descriptor,
        'mode': mode
    }

    # Training loop
    print(f"\nStarting training for {TRAIN_CONFIG['num_epochs']} epochs...")
    print("-" * 70)

    for epoch in range(TRAIN_CONFIG['num_epochs']):
        epoch_loss = 0
        num_batches = 0

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{TRAIN_CONFIG['num_epochs']}")

        for batch_idx, (embeddings, labels) in enumerate(pbar):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            projected = model(embeddings)

            if miner:
                hard_pairs = miner(projected, labels)
                loss = loss_fn(projected, labels, hard_pairs)
            else:
                loss = loss_fn(projected, labels)

            if not torch.isfinite(loss) or loss.item() < 1e-8:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })

        avg_train_loss = epoch_loss / max(num_batches, 1)
        history['train_loss'].append(avg_train_loss)
        history['learning_rates'].append(scheduler.get_last_lr()[0])

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        # Validation
        if (epoch + 1) % TRAIN_CONFIG['validate_every_n_epochs'] == 0:
            val_metrics = validate_retrieval_accuracy(model, val_loader, device, train_loader)
            val_accuracy = val_metrics['accuracy']
            history['val_accuracy'].append(val_accuracy)
            history.setdefault('val_recall@1', []).append(val_metrics['recall@1'])
            history.setdefault('val_recall@5', []).append(val_metrics['recall@5'])
            history.setdefault('val_recall@10', []).append(val_metrics['recall@10'])
            history.setdefault('val_mrr', []).append(val_metrics['mrr'])

            print(f"  Val Recall@1: {val_metrics['recall@1']:.4f}")
            print(f"  Val Recall@5: {val_metrics['recall@5']:.4f}")
            print(f"  Val Recall@10: {val_metrics['recall@10']:.4f}")
            print(f"  Val MRR: {val_metrics['mrr']:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0

                # Explicit model architecture config for reliable loading
                model_config = {
                    'input_dim': TRAIN_CONFIG['input_dim'],
                    'hidden_dim': TRAIN_CONFIG['hidden_dim'],
                    'output_dim': TRAIN_CONFIG['output_dim'],
                    'num_hidden_layers': TRAIN_CONFIG['num_hidden_layers'],
                    'dropout': TRAIN_CONFIG['dropout'],
                }

                best_model_name = f'best_{model_descriptor}_acc{val_accuracy:.4f}.pth'
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'model_config': model_config,  # Explicit architecture params
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_accuracy': val_accuracy,
                    'train_loss': avg_train_loss,
                    'config': {k: str(v) if isinstance(v, Path) else v
                               for k, v in TRAIN_CONFIG.items()},
                    'model_descriptor': model_descriptor,
                    'dataset_info': {
                        'num_train': len(train_dataset),
                        'num_val': len(val_dataset),
                        'num_classes': len(train_dataset.class_to_idx),
                        'mode': mode
                    }
                }

                save_path = save_dir / best_model_name
                torch.save(checkpoint, save_path)
                print(f"  [SAVED] New best model: {best_model_name}")

                torch.save(checkpoint, save_dir / 'vision_model.pth')
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{TRAIN_CONFIG['patience']}")

        if patience_counter >= TRAIN_CONFIG['patience']:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation accuracy: {best_val_accuracy:.4f}")
            break

    # Save final model
    model_config = {
        'input_dim': TRAIN_CONFIG['input_dim'],
        'hidden_dim': TRAIN_CONFIG['hidden_dim'],
        'output_dim': TRAIN_CONFIG['output_dim'],
        'num_hidden_layers': TRAIN_CONFIG['num_hidden_layers'],
        'dropout': TRAIN_CONFIG['dropout'],
    }

    final_model_name = f'final_{model_descriptor}_epochs{epoch + 1}_acc{best_val_accuracy:.4f}.pth'
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'history': history,
        'config': {k: str(v) if isinstance(v, Path) else v
                   for k, v in TRAIN_CONFIG.items()},
        'model_descriptor': model_descriptor,
        'final_epoch': epoch + 1,
        'best_val_accuracy': best_val_accuracy,
        'mode': mode
    }

    torch.save(final_checkpoint, save_dir / final_model_name)

    # Save training history
    history_name = f'history_{model_descriptor}.json'
    with open(save_dir / history_name, 'w') as f:
        json.dump(history, f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print(f"Mode: {mode.upper()}")
    print(f"Configuration: {model_descriptor}")
    print(f"Total epochs: {epoch + 1}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Models saved to: {save_dir}")
    print(f"  Best model: best_{model_descriptor}_acc{best_val_accuracy:.4f}.pth")
    print(f"  Final model: {final_model_name}")
    print("=" * 70)

    return model, history


def main():
    """Entry point"""
    return train(mode=TRAINING_MODE)


if __name__ == "__main__":
    main()
