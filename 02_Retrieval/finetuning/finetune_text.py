"""
Fine-tuning text embeddings for meme retrieval
Optimized for noisy OCR data with text augmentation
Follows the same structure as finetune_siglip.py
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

# Use config paths
TEXT_EMBEDDINGS_DIR = RETRIEVAL_CONFIG.TEXT_EMBEDDINGS_DIR
TEXT_METADATA_DIR = RETRIEVAL_CONFIG.TEXT_METADATA_DIR
TEXT_INDEX_DIR = RETRIEVAL_CONFIG.TEXT_INDEX_DIR
# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_CONFIG = {
    # Data parameters
    'batch_size': 9028,
    'val_batch_size': 9028,
    'num_epochs': 200,
    'random_seed': 42,

    # FILTERING - Confirmed only, with quality thresholds
    'use_only_confirmed': True,
    'min_text_length': 3,  # Minimum words for training
    'min_char_length': 10,  # Minimum characters

    # Model parameters
    'input_dim': 768,
    'hidden_dim': 3072,
    'output_dim': 768,
    'num_hidden_layers': 3,
    'dropout': 0.2,

    # Optimizer parameters
    'learning_rate': 1e-4,  # Lower LR for noisy data
    'weight_decay': 0.01,

    'loss_type': 'SupConLoss',
    'temperature': 0.07,
    'miner_type': None,
    'miner_epsilon': 0.1,  # Not used for TripletMarginMiner
    'miner_margin': 0.7,
    'miner_triplet_type': 'semihard',

    # Text augmentation parameters
    'augmentation_prob': 0.3,  # Probability of augmenting each sample
    'token_dropout_prob': 0.1,  # Probability of dropping each token
    'char_noise_prob': 0.05,  # Probability of character-level noise

    # System parameters
    'num_workers': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Saving parameters - matches image model structure
    'save_dir': RETRIEVAL_CONFIG.OUTPUT_DIR / 'finetuned_models' / 'text',
    'save_checkpoints': False,  # No intermediate checkpoints
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
        """Random case changes to handle meme text variations"""
        if random.random() < 0.5:
            return text.upper()
        elif random.random() < 0.5:
            return text.lower()
        else:
            # Random mixed case
            return ''.join(c.upper() if random.random() < 0.5 else c.lower() for c in text)

    @staticmethod
    def token_dropout(text, dropout_prob=0.1):
        """Randomly drop tokens to simulate incomplete OCR"""
        tokens = text.split()
        if len(tokens) <= 2:  # Don't dropout very short texts
            return text
        kept_tokens = [t for t in tokens if random.random() > dropout_prob]
        return ' '.join(kept_tokens) if kept_tokens else text

    @staticmethod
    def char_noise(text, noise_prob=0.05):
        """Add character-level noise to simulate OCR errors"""
        if not text:
            return text

        chars = list(text)
        for i in range(len(chars)):
            if random.random() < noise_prob:
                # Random char substitution/deletion
                if random.random() < 0.5 and i > 0:
                    # Swap with adjacent
                    chars[i], chars[i - 1] = chars[i - 1], chars[i]
                elif random.random() < 0.3:
                    # Delete (replace with space)
                    chars[i] = ' '

        # Clean up multiple spaces
        result = ''.join(chars)
        result = re.sub(r'\s+', ' ', result)
        return result.strip()

    @staticmethod
    def augment(text, config):
        """Apply random augmentation based on config"""
        if random.random() > config['augmentation_prob']:
            return text  # No augmentation

        # Apply one or more augmentations
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

class ConfirmedTextEmbeddingDataset(Dataset):
    """Dataset for fine-tuning text embeddings with quality filtering"""

    def __init__(self, split='train', max_chunks=None, apply_augmentation=True):
        self.split = split
        self.apply_augmentation = apply_augmentation and (split == 'train')
        self.samples = []
        self.class_to_idx = {}

        print(f"\n{'=' * 60}")
        print(f"Loading {split.upper()} Text Dataset - CONFIRMED ONLY")
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
        filtered_texts = 0

        # Load chunks and filter
        for emb_file, meta_file in tqdm(zip(embedding_files, metadata_files),
                                        total=len(embedding_files),
                                        desc="Loading and filtering"):
            # Load embeddings
            chunk_embeddings = np.load(emb_file)

            # Load metadata
            with open(meta_file, 'r') as f:
                chunk_metadata = json.load(f)

            # Filter: Only confirmed texts with sufficient quality
            for i, text_meta in enumerate(chunk_metadata['valid_texts']):
                total_texts += 1

                # Check if confirmed
                dataset_type = text_meta.get('dataset_type', 'unknown')
                if dataset_type != 'confirmed':
                    continue

                # Quality checks
                text = text_meta.get('text', '')
                word_count = text_meta.get('word_count', 0)
                text_length = text_meta.get('text_length', 0)

                # Apply minimum thresholds
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

                confirmed_texts += 1
                all_embeddings.append(chunk_embeddings[i])
                all_metadata.append(text_meta)

        print(f"\nFiltering Results:")
        print(f"  Total texts scanned: {total_texts:,}")
        print(f"  Confirmed texts found: {confirmed_texts:,}")
        print(f"  Texts filtered (too short/low quality): {filtered_texts:,}")

        if confirmed_texts == 0:
            raise ValueError("No confirmed texts passed quality filters!")

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
                # Deterministic split
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

        # Apply augmentation during training
        if self.apply_augmentation:
            # Note: We augment the text conceptually, but since we're using
            # pre-computed embeddings, we'll add small noise to embeddings
            # to simulate text variations
            if random.random() < TRAIN_CONFIG['augmentation_prob']:
                # Add small Gaussian noise to embedding
                noise = np.random.normal(0, 0.01, embedding.shape)
                embedding = embedding + noise
                # Re-normalize
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return torch.from_numpy(embedding).float(), label


# ============================================================================
# MODEL
# ============================================================================

class TextProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=3072, output_dim=768,
                 num_hidden_layers=3, dropout=0.2):
        super().__init__()

        # Build layers based on num_hidden_layers
        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.Sequential(*layers)
        self.residual_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        transformed = self.layers(x)
        out = x * (1 - self.residual_weight) + transformed * self.residual_weight
        return F.normalize(out, p=2, dim=1)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def get_model_descriptor(config):
    """Create descriptive string for model configuration"""
    loss_map = {
        'NTXentLoss': 'NTXent',
        'ContrastiveLoss': 'Contrastive',
        'MultiSimilarityLoss': 'MSLoss',
        'SupConLoss': 'SupConLoss'
    }

    loss_short = loss_map.get(config['loss_type'], config['loss_type'])

    descriptor = f"Text_{loss_short}_{config['hidden_dim']}d"

    if config['augmentation_prob'] > 0:
        descriptor += "_aug"

    return descriptor


def get_loss_and_miner(config):
    """Initialize loss function and miner for text training"""

    distance = distances.CosineSimilarity()

    if config['loss_type'] == 'SupConLoss':
        loss_fn = losses.SupConLoss(
            temperature=0.05,  # Lower temperature often better
            distance=distance
        )
    # Loss function - NTXent recommended for noisy text
    elif config['loss_type'] == 'NTXentLoss':
        loss_fn = losses.NTXentLoss(
            temperature=config['temperature'],
            distance=distance
        )
    elif config['loss_type'] == 'ContrastiveLoss':
        loss_fn = losses.ContrastiveLoss(
            pos_margin=0.0,
            neg_margin=1.0,
            distance=distance
        )
    elif config['loss_type'] == 'MultiSimilarityLoss':
        loss_fn = losses.MultiSimilarityLoss(
            alpha=2, beta=50, base=0.5,
            distance=distance
        )
    else:
        raise ValueError(f"Unknown loss type: {config['loss_type']}")

    # Miner (optional for text)
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

    return loss_fn, miner


def validate_retrieval_accuracy_with_faiss(model, train_loader, val_loader, device):
    """Validation using FAISS indices"""
    import faiss

    model.eval()

    # Collect train embeddings
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

    # Build FAISS index
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

    # Search
    print(f"  Searching {len(val_embeddings)} queries...", end="")
    k = 1
    distances, indices = train_index.search(val_embeddings, k)
    print(" Done")

    # Calculate accuracy
    predicted_labels = train_labels[indices.squeeze()]
    correct = (predicted_labels == val_labels).sum()
    accuracy = correct / len(val_labels)

    model.train()
    return accuracy


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_text():
    """Main training function for text embeddings"""

    # Get model descriptor
    model_descriptor = get_model_descriptor(TRAIN_CONFIG)

    print("\n" + "=" * 70)
    print("FINE-TUNING")
    print(f"Configuration: {model_descriptor}")
    print("=" * 70)

    device = torch.device(TRAIN_CONFIG['device'])
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directory
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
    train_dataset = ConfirmedTextEmbeddingDataset(split='train', apply_augmentation=True)
    val_dataset = ConfirmedTextEmbeddingDataset(split='val', apply_augmentation=False)

    print(f"\n{'=' * 60}")
    print(f"Dataset Summary:")
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

    # Optimizer - lower learning rate for noisy data
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

            # Skip invalid losses
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
            val_accuracy = validate_retrieval_accuracy_with_faiss(
                model, train_loader, val_loader, device
            )
            history['val_accuracy'].append(val_accuracy)

            print(f"  Val Accuracy: {val_accuracy:.4f}")

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0

                # Save with descriptive name
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
                        'confirmed_only': True,
                        'min_text_length': TRAIN_CONFIG['min_text_length']
                    }
                }

                save_path = save_dir / best_model_name
                torch.save(checkpoint, save_path)
                print(f"  [SAVED] New best model: {best_model_name}")

                # Also save as 'best_model.pth' for easy loading
                torch.save(checkpoint, save_dir / 'text_model.pth')
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{TRAIN_CONFIG['patience']}")

        # Early stopping
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
        'best_val_accuracy': best_val_accuracy
    }

    torch.save(final_checkpoint, save_dir / final_model_name)

    # Save training history
    history_name = f'history_text_{model_descriptor}.json'
    with open(save_dir / history_name, 'w') as f:
        json.dump(history, f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print(f"Configuration: {model_descriptor}")
    print(f"Total epochs: {epoch + 1}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Models saved to: {save_dir}")
    print(f"  Best model: best_text_{model_descriptor}_acc{best_val_accuracy:.4f}.pth")
    print(f"  Final model: {final_model_name}")
    print(f"  Update models.py with new configuration")
    print("=" * 70)

    return model, history


def main():
    """Entry point"""
    return train_text()


if __name__ == "__main__":
    main()