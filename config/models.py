# models.py - CORRECTED VERSION
# Matches the exact architectures from finetune_siglip.py and finetune_text.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from sentence_transformers import SentenceTransformer
from pathlib import Path
from PIL import Image
import numpy as np


class ProjectionHead(nn.Module):
    """Vision projection head - exactly as in finetune_siglip.py"""

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


class TextProjectionHead(nn.Module):
    """Text projection head - exactly as in finetune_text.py"""

    def __init__(self, input_dim=768, hidden_dim=3072, output_dim=768,  # Changed from 1024 to 3072
                 num_hidden_layers=3, dropout=0.2):  # Changed from 2 to 3
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


class SigLIPWithProjection:
    """Complete vision model: SigLIP base + trained projection"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("Loading SigLIP base model...")
        self.model = AutoModel.from_pretrained('google/siglip-base-patch16-384')
        self.processor = AutoProcessor.from_pretrained('google/siglip-base-patch16-384')
        self.model.to(self.device)
        self.model.eval()

        self.projection = None

    def load_finetuned(self, checkpoint_path):
        """Load your trained projection head"""
        if Path(checkpoint_path).exists():
            print(f"Loading vision projection from {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # CORRECTED: Use ProjectionHead, not VisionProjectionHead
            self.projection = ProjectionHead(
                input_dim=checkpoint['config'].get('input_dim', 768),
                hidden_dim=checkpoint['config'].get('hidden_dim', 3072),
                output_dim=checkpoint['config'].get('output_dim', 768),
                num_hidden_layers=checkpoint['config'].get('num_hidden_layers', 3),
                dropout=0.0  # No dropout during inference
            )

            self.projection.load_state_dict(checkpoint['model_state_dict'])
            self.projection.to(self.device)
            self.projection.eval()

            # Display loaded model info
            if 'val_accuracy' in checkpoint:
                print(f"Loaded vision model with accuracy: {checkpoint['val_accuracy']:.4f}")
            elif 'best_val_accuracy' in checkpoint:
                print(f"Loaded vision model with accuracy: {checkpoint['best_val_accuracy']:.4f}")
            else:
                print("Loaded vision model (no accuracy info in checkpoint)")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

    def encode_image(self, image_path):
        """Encode image: SigLIP + projection"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.vision_model(**inputs)
            base_embedding = outputs.pooler_output

            if self.projection is not None:
                final_embedding = self.projection(base_embedding)
            else:
                final_embedding = F.normalize(base_embedding, p=2, dim=1)

        return final_embedding.cpu().numpy()


class SentenceTransformerWithProjection:
    """Complete text model: Sentence transformer base + trained projection"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("Loading sentence transformer base model...")
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        self.projection = None

    def load_finetuned(self, checkpoint_path):
        """Load your trained projection head"""
        if Path(checkpoint_path).exists():
            print(f"Loading text projection from {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Use TextProjectionHead with correct parameters
            self.projection = TextProjectionHead(
                input_dim=checkpoint['config'].get('input_dim', 768),
                hidden_dim=checkpoint['config'].get('hidden_dim', 3072),
                output_dim=checkpoint['config'].get('output_dim', 768),
                num_hidden_layers=checkpoint['config'].get('num_hidden_layers', 3),
                dropout=0.0
            )

            self.projection.load_state_dict(checkpoint['model_state_dict'])
            self.projection.to(self.device)
            self.projection.eval()

            # Display loaded model info
            if 'val_accuracy' in checkpoint:
                print(f"Loaded text model with accuracy: {checkpoint['val_accuracy']:.4f}")
            elif 'best_val_accuracy' in checkpoint:
                print(f"Loaded text model with accuracy: {checkpoint['best_val_accuracy']:.4f}")
            else:
                print("Loaded text model (no accuracy info in checkpoint)")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")

    def encode_text(self, text):
        """Encode text: Sentence transformer + projection"""
        base_embedding = self.model.encode(text, convert_to_tensor=True, device=str(self.device))

        with torch.no_grad():
            if base_embedding.dim() == 1:
                base_embedding = base_embedding.unsqueeze(0)

            if self.projection is not None:
                final_embedding = self.projection(base_embedding)
            else:
                final_embedding = F.normalize(base_embedding, p=2, dim=1)

        return final_embedding.squeeze(0).cpu().numpy()


def load_vision_model(model_path=None, device='cuda'):
    """Load complete vision model"""
    model = SigLIPWithProjection()
    if model_path:
        model.load_finetuned(model_path)
    return model


def load_text_model(model_path=None, device='cuda'):
    """Load complete text model"""
    model = SentenceTransformerWithProjection()
    if model_path:
        model.load_finetuned(model_path)
    return model


def load_unified_models(vision_path=None, text_path=None, device='cuda'):
    """Load both models"""
    vision_model = load_vision_model(vision_path, device)
    text_model = load_text_model(text_path, device)

    return {
        'vision': vision_model,
        'text': text_model
    }