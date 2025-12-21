"""
Fine-tuning text embeddings with configurable data source
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
import re
import sys

# PyTorch Metric Learning
from pytorch_metric_learning import distances, losses, miners

sys.path.append(str(Path(__file__).parent.parent))
from config.config import RETRIEVAL_CONFIG

TEXT_EMBEDDINGS_DIR = RETRIEVAL_CONFIG.TEXT_EMBEDDINGS_DIR
TEXT_METADATA_DIR = RETRIEVAL_CONFIG.TEXT_METADATA_DIR
TEXT_INDEX_DIR = RETRIEVAL_CONFIG.TEXT_INDEX_DIR

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
    return RETRIEVAL_CONFIG.OUTPUT_DIR / 'finetuned_models' / 'text' / mode


TRAIN_CONFIG = {
    # Data parameters
    'batch_size': 9028,
    'val_batch_size': 9028,
    'num_epochs': 10,
    'random_seed': 42,

    # Quality thresholds
    'min_text_length': 3,  # Minimum words
    'min_char_length': 10,  # Minimum characters

    # Model parameters
    'input_dim': 768,
    'hidden_dim': 3072,
    'output_dim': 768,
    'num_hidden_layers': 3,
    'dropout': 0.2,

    # Optimizer parameters
    'learning_rate': 1e-4,
    'weight_decay': 0.01,

    'loss_type': 'SupConLoss',
    'temperature': 0.07,
    'miner_type': None,
    'miner_epsilon': 0.1,
    'miner_margin': 0.7,
    'miner_triplet_type': 'semihard',

    # Text augmentation parameters
    'augmentation_prob': 0.3,
    'token_dropout_prob': 0.1,
    'char_noise_prob': 0.05,

    # System parameters
    'num_workers': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Saving parameters
    'save_dir': get_save_dir(TRAINING_MODE),
    'save_checkpoints': False,
    'patience': 15,

    # Validation
    'validate_every_n_epochs': 1,
}


# ============================================================================
# TEXT AUGMENTATION
# ============================================================================

class TextAugmenter:
    """Augmentation strategies for noisy OCR text"""

    @staticmethod
    def random_case(text):
        if random.random() < 0.5:
            return text.upper()
        elif random.random() < 0.5:
            return text.lower()
        else:
            return ''.join(c.upper() if random.random() < 0.5 else c.lower() for c in text)

    @staticmethod
    def token_dropout(text, dropout_prob=0.1):
        tokens = text.split()
        if len(tokens) <= 2:
            return text
        kept_tokens = [t for t in tokens if random.random() > dropout_prob]
        return ' '.join(kept_tokens) if kept_tokens else text

    @staticmethod
    def char_noise(text, noise_prob=0.05):
        if not text:
            return text

        chars = list(text)
        for i in range(len(chars)):
            if random.random() < noise_prob:
                if random.random() < 0.5 and i > 0:
                    chars[i], chars[i - 1] = chars[i - 1], chars[i]
                elif random.random() < 0.3:
                    chars[i] = ' '

        result = ''.join(chars)
        result = re.sub(r'\s+', ' ', result)
        return result.strip()

    @staticmethod
    def augment(text, config):
        if random.random() > config['augmentation_prob']:
            return text

        if random.random() < 0.5:
            text = TextAugmenter.random_case(text)

        if random.random() < 0.5:
            text = TextAugmenter.token_dropout(text, config['token_dropout_prob'])

        if random.random() < 0.3:
            text = TextAugmenter.char_noise(text, config['char_noise_prob'])

        return text


# ============================================================================
# DATASET
# ============================================================================

class TextEmbeddingDataset(Dataset):
    """Dataset for fine-tuning text embeddings based on training mode"""

    def __init__(self, split='train', mode='confirmed', max_chunks=None, apply_augmentation=True):
        self.split = split
        self.mode = mode
        self.apply_augmentation = apply_augmentation and (split == 'train')
        self.samples = []
        self.class_to_idx = {}

        print(f"\n{'=' * 60}")
        print(f"Loading {split.upper()} Text Dataset - MODE: {mode.upper()}")
        print('=' * 60)

        # Get all chunk files
        embedding_files = sorted(TEXT_EMBEDDINGS_DIR.glob("text_embeddings_chunk_*.npy"))
        metadata_files = sorted(TEXT_METADATA_DIR.glob("text_metadata_chunk_*.json"))

        if not embedding_files:
            raise FileNotFoundError(f"No text embeddings found in {TEXT_EMBEDDINGS_DIR}")

        if max_chunks:
            embedding_files = embedding_files[:max_chunks]
            metadata_files = metadata_files[:max_chunks]

        print(f"Processing {len(embedding_files)} chunks...")

        # Storage
        all_embeddings = []
        all_metadata = []

        # Statistics
        total_texts = 0
        confirmed_texts = 0
        unconfirmed_texts = 0
        included_texts = 0
        filtered_texts = 0

        # Load chunks and filter
        for emb_file, meta_file in tqdm(zip(embedding_files, metadata_files),
                                        total=len(embedding_files),
                                        desc="Loading and filtering"):
            chunk_embeddings = np.load(emb_file)

            with open(meta_file, 'r') as f:
                chunk_metadata = json.load(f)

            for i, text_meta in enumerate(chunk_metadata['valid_texts']):
                total_texts += 1

                # Determine text type from metadata
                dataset_type = text_meta.get('dataset_type', 'unknown')
                is_confirmed = dataset_type == 'confirmed'
                is_unconfirmed = dataset_type == 'unconfirmed'

                if is_confirmed:
                    confirmed_texts += 1
                elif is_unconfirmed:
                    unconfirmed_texts += 1

                # Filter based on mode
                include = False
                if mode == 'confirmed' and is_confirmed:
                    include = True
                elif mode == 'unconfirmed' and is_unconfirmed:
                    include = True
                elif mode == 'combined' and (is_confirmed or is_unconfirmed):
                    include = True

                if not include:
                    continue

                # Quality checks
                text = text_meta.get('text', '')
                word_count = text_meta.get('word_count', 0)
                text_length = text_meta.get('text_length', 0)

                if word_count < TRAIN_CONFIG['min_text_length']:
                    filtered_texts += 1
                    continue

                if text_length < TRAIN_CONFIG['min_char_length']:
                    filtered_texts += 1
                    continue

                # Skip texts that are just numbers or special chars
                if not re.search(r'[a-zA-Z]{2,}', text):
                    filtered_texts += 1
                    continue

                included_texts += 1
                all_embeddings.append(chunk_embeddings[i])
                all_metadata.append(text_meta)

        print(f"\nFiltering Results:")
        print(f"  Total texts scanned: {total_texts:,}")
        print(f"  Confirmed texts: {confirmed_texts:,}")
        print(f"  Unconfirmed texts: {unconfirmed_texts:,}")
        print(f"  Filtered (too short/low quality): {filtered_texts:,}")
        print(f"  Included for training ({mode}): {included_texts:,}")

        if included_texts == 0:
            raise ValueError(f"No texts passed quality filters for mode: {mode}")

        # Stack embeddings
        self.embeddings = np.vstack(all_embeddings)
        print(f"\nLoaded embeddings shape: {self.embeddings.shape}")

        # Build class mapping
        unique_classes = set(meta['class_id'] for meta in all_metadata)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        print(f"Unique classes: {len(self.class_to_idx)}")

        # Group by class for stratified splitting
        class_to_samples = defaultdict(list)
        for idx, meta in enumerate(all_metadata):
            class_id = meta['class_id']
            class_to_samples[class_id].append({
                'embedding_idx': idx,
                'text': meta.get('text', ''),
                'meta': meta
            })

        # Split 80/20 per class
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
                        'text': sample_data['text']
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
                        'text': sample_data['text']
                    })

        print(f"\n{split.upper()} split: {len(self.samples)} samples")

        # Text length statistics
        text_lengths = [len(s['text'].split()) for s in self.samples]
        print(f"Text statistics:")
        print(f"  Min words: {min(text_lengths)}")
        print(f"  Max words: {max(text_lengths)}")
        print(f"  Avg words: {np.mean(text_lengths):.1f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        embedding = self.embeddings[sample['embedding_idx']]
        label = sample['label']

        if self.apply_augmentation:
            if random.random() < TRAIN_CONFIG['augmentation_prob']:
                noise = np.random.normal(0, 0.01, embedding.shape)
                embedding = embedding + noise
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return torch.from_numpy(embedding).float(), label


# ============================================================================
# MODEL
# ============================================================================

class TextProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=3072, output_dim=768,
                 num_hidden_layers=3, dropout=0.2):
        super().__init__()

        layers = []

        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])

        # Middle layers
        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        # Final projection
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        ])

        self.layers = nn.Sequential(*layers)
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
            temperature=config['temperature'],
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


def validate_retrieval_accuracy_with_faiss(model, train_loader, val_loader, device):
    """Validate using retrieval accuracy, recall, and MRR"""
    model.eval()

    # Collect all training embeddings for reference
    train_embeddings = []
    train_labels = []

    with torch.no_grad():
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            projected = model(embeddings)
            train_embeddings.append(projected.cpu())
            train_labels.append(labels)

    train_embeddings = torch.cat(train_embeddings, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # Collect validation embeddings
    val_embeddings = []
    val_labels = []

    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings = embeddings.to(device)
            projected = model(embeddings)
            val_embeddings.append(projected.cpu())
            val_labels.append(labels)

    val_embeddings = torch.cat(val_embeddings, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    # Compute similarities between val and train
    similarities = torch.mm(val_embeddings, train_embeddings.t())

    # Get top-k indices
    k_max = min(10, similarities.size(1))
    _, top1_indices = similarities.topk(1, dim=1)
    _, top5_indices = similarities.topk(min(5, k_max), dim=1)
    _, top10_indices = similarities.topk(k_max, dim=1)

    # Recall@1
    predictions = train_labels[top1_indices.squeeze()]
    recall_1 = (predictions == val_labels).float().mean().item()

    # Recall@5 and Recall@10
    recall_5 = 0
    recall_10 = 0
    for i in range(len(val_labels)):
        if val_labels[i] in train_labels[top5_indices[i]]:
            recall_5 += 1
        if val_labels[i] in train_labels[top10_indices[i]]:
            recall_10 += 1

    recall_5 /= len(val_labels)
    recall_10 /= len(val_labels)

    # MRR - Mean Reciprocal Rank
    # For each val query, find rank of first correct match in train set
    _, all_indices = similarities.topk(similarities.size(1), dim=1)
    reciprocal_ranks = []
    for i in range(len(val_labels)):
        query_label = val_labels[i]
        retrieved_labels = train_labels[all_indices[i]]
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

def train_text(mode=None):
    """Main training function for text embeddings"""
    if mode is None:
        mode = TRAINING_MODE

    print(f"\n{'=' * 70}")
    print(f"TEXT EMBEDDING FINE-TUNING")
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
    train_dataset = TextEmbeddingDataset(split='train', mode=mode, apply_augmentation=True)
    val_dataset = TextEmbeddingDataset(split='val', mode=mode, apply_augmentation=False)

    print(f"\n{'=' * 60}")
    print(f"Dataset Summary:")
    print(f"  Mode: {mode}")
    print(f"  Train: {len(train_dataset)} samples (with augmentation)")
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
    print("\nInitialising text projection head...")
    model = TextProjectionHead(
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
            val_metrics = validate_retrieval_accuracy_with_faiss(
                model, train_loader, val_loader, device
            )
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

                best_model_name = f'best_text_{model_descriptor}_acc{val_accuracy:.4f}.pth'
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
                        'mode': mode,
                        'min_text_length': TRAIN_CONFIG['min_text_length']
                    }
                }

                save_path = save_dir / best_model_name
                torch.save(checkpoint, save_path)
                print(f"  [SAVED] New best model: {best_model_name}")

                torch.save(checkpoint, save_dir / 'text_model.pth')
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{TRAIN_CONFIG['patience']}")

        if patience_counter >= TRAIN_CONFIG['patience']:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation accuracy: {best_val_accuracy:.4f}")
            break

    # Save final model
    final_model_name = f'final_text_{model_descriptor}_epochs{epoch + 1}_acc{best_val_accuracy:.4f}.pth'
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': TRAIN_CONFIG,
        'model_descriptor': model_descriptor,
        'final_epoch': epoch + 1,
        'best_val_accuracy': best_val_accuracy,
        'mode': mode
    }

    torch.save(final_checkpoint, save_dir / final_model_name)

    # Save training history
    history_name = f'history_text_{model_descriptor}.json'
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
    print(f"  Best model: best_text_{model_descriptor}_acc{best_val_accuracy:.4f}.pth")
    print(f"  Final model: {final_model_name}")
    print("=" * 70)

    return model, history


def main():
    """Entry point"""
    return train_text(mode=TRAINING_MODE)


if __name__ == "__main__":
    main()
