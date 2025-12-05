#!/usr/bin/env python3
"""
Similarity Search with Proper Projection Heads
==============================================
Uses the same projection heads that were used to build the FAISS indices.
"""

# Suppress warnings
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP conflict

import warnings
import torchvision

torchvision.disable_beta_transforms_warning()  # Silence torchvision beta warning
warnings.filterwarnings("ignore", message=".*hf_xet.*")  # Silence HF Xet storage warning

import json
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rdflib import Graph, Namespace
from rdflib.namespace import RDF, RDFS

# Import the models
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from sentence_transformers import SentenceTransformer

# Import your config
from config.namespaces import SMO, EX, SCHEMA, OWL
from config.config import META_CONFIG, CORPUS_CONFIG, RETRIEVAL_CONFIG

# CRITICAL: Import projection heads from models.py
from config.models import SigLIPWithProjection, SentenceTransformerWithProjection

# Paths from config (using YOUR paths)
corpus_graph = CORPUS_CONFIG.INSTANCE_GRAPH_PATH
vision_index = CORPUS_CONFIG.VISION_INDEX_PATH
text_index = CORPUS_CONFIG.TEXT_INDEX_PATH
vision_model = CORPUS_CONFIG.VISION_MODEL_PATH
text_model = CORPUS_CONFIG.TEXT_MODEL_PATH

# Output and dataset paths
output_dir = [UPDATE PATH]
dataset_base_dir =  [UPDATE PATH]


class SimilaritySearch:
    """Simple similarity search using FAISS indices with projection heads"""

    def __init__(
            self,
            vision_index_path: str = None,
            text_index_path: str = None,
            vision_model_path: str = None,
            text_model_path: str = None,
            rdf_graph_path: str = None,
            dataset_base_dir: str = None,
            search_mode: str = 'all'
    ):
        """
        Initialize the searcher with FAISS indices and projection heads

        Args:
            vision_index_path: Path to FAISS vision index
            text_index_path: Path to FAISS text index
            vision_model_path: Path to vision model checkpoint with projection
            text_model_path: Path to text model checkpoint with projection
            rdf_graph_path: Path to RDF graph with bridge mappings
            dataset_base_dir: Base directory for meme images
            search_mode: 'vision', 'text', 'multimodal', or 'all'
        """
        self.search_mode = search_mode
        self.dataset_base_dir = Path(dataset_base_dir) if dataset_base_dir else None
        # Remove device - models.py handles this internally

        print(f"\nInitializing Similarity Searcher")
        print(f"Search mode: {search_mode.upper()}")
        print("=" * 60)

        # Initialize components
        self.vision_index = None
        self.text_index = None
        self.vision_model = None  # Will be SigLIPWithProjection from models.py
        self.text_model = None  # Will be SentenceTransformerWithProjection from models.py
        self.rdf_graph = None

        # Mappings from FAISS indices to RDF instances
        self.vision_idx_to_instance = {}
        self.text_idx_to_instance = {}

        # Load RDF graph if provided
        if rdf_graph_path and Path(rdf_graph_path).exists():
            self._load_rdf_graph(rdf_graph_path)

        # Conditional loading based on search mode
        if search_mode in ['vision', 'multimodal', 'all']:
            if vision_index_path and Path(vision_index_path).exists():
                self._load_vision_components(vision_index_path, vision_model_path)

        if search_mode in ['text', 'multimodal', 'all']:
            if text_index_path and Path(text_index_path).exists():
                self._load_text_components(text_index_path, text_model_path)

        print("\nSearcher ready!")

    def analyse_distance_distribution(self, image_path: str, text_query: str, k: int = 1000):
        """Analyse distance distributions for a query."""
        vision_results = self.search_by_image(image_path, k=k)
        text_results = self.search_by_text(text_query, k=k)

        v_dists = [r['distance'] for r in vision_results]
        t_dists = [r['distance'] for r in text_results]

        print("VISION distances:")
        print(f"  Min: {min(v_dists):.4f}, Max: {max(v_dists):.4f}")
        print(f"  Rank 10: {v_dists[9]:.4f}, Rank 100: {v_dists[99]:.4f}, Rank 500: {v_dists[499]:.4f}")
        print(f"  Mean: {np.mean(v_dists):.4f}, Std: {np.std(v_dists):.4f}")

        print("TEXT distances:")
        print(f"  Min: {min(t_dists):.4f}, Max: {max(t_dists):.4f}")
        print(f"  Rank 10: {t_dists[9]:.4f}, Rank 100: {t_dists[99]:.4f}, Rank 500: {t_dists[499]:.4f}")
        print(f"  Mean: {np.mean(t_dists):.4f}, Std: {np.std(t_dists):.4f}")

        return v_dists, t_dists

    def _load_rdf_graph(self, rdf_graph_path: str):
        """Load RDF graph and build FAISS to instance mappings"""
        print(f"\nLoading RDF graph from {rdf_graph_path}")
        self.rdf_graph = Graph()
        self.rdf_graph.parse(rdf_graph_path, format="turtle")
        print(f"  Loaded {len(self.rdf_graph)} triples")

        # Build FAISS index to RDF instance mappings
        self._build_faiss_mappings()

    def _build_faiss_mappings(self):
        """Build mappings from FAISS indices to RDF instances"""
        print("  Building FAISS to RDF mappings...")

        # Query for vision indices
        vision_query = """
        PREFIX smo: <http://semiomeme.org/ontology/>
        SELECT ?instance ?idx WHERE {
            ?instance smo:faissVisionIndex ?idx .
        }
        """
        for row in self.rdf_graph.query(vision_query):
            idx = int(row.idx)
            self.vision_idx_to_instance[idx] = row.instance

        # Query for text indices
        text_query = """
        PREFIX smo: <http://semiomeme.org/ontology/>
        SELECT ?instance ?idx WHERE {
            ?instance smo:faissTextIndex ?idx .
        }
        """
        for row in self.rdf_graph.query(text_query):
            idx = int(row.idx)
            self.text_idx_to_instance[idx] = row.instance

        print(f"    Vision mappings: {len(self.vision_idx_to_instance)}")
        print(f"    Text mappings: {len(self.text_idx_to_instance)}")

    def _load_vision_components(self, index_path: str, model_path: str):
        """Load vision index and model using models.py"""
        print(f"\nLoading vision components...")

        # Load FAISS index
        self.vision_index = faiss.read_index(index_path)
        print(f"  Vision index: {self.vision_index.ntotal} vectors")
        self.vision_index.nprobe = 32

        # Load model from models.py
        self.vision_model = SigLIPWithProjection()

        # Load finetuned projection if available
        if model_path and Path(model_path).exists():
            self.vision_model.load_finetuned(model_path)
        else:
            print("  WARNING: No vision projection checkpoint - using base SigLIP only")

    def _load_text_components(self, index_path: str, model_path: str):
        """Load text index and model using models.py"""
        print(f"\nLoading text components...")

        # Load FAISS index
        self.text_index = faiss.read_index(index_path)
        print(f"  Text index: {self.text_index.ntotal} vectors")
        self.text_index.nprobe = 32

        # Load model from models.py
        self.text_model = SentenceTransformerWithProjection()

        # Load finetuned projection if available
        if model_path and Path(model_path).exists():
            self.text_model.load_finetuned(model_path)
        else:
            print("  WARNING: No text projection checkpoint - using base Sentence-BERT only")

    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode image using the model from models.py"""
        if not self.vision_model:
            raise ValueError("Vision model not loaded")

        # Use the model's built-in encode method
        embedding = self.vision_model.encode_image(image_path)

        # Ensure it's 2D for FAISS (models.py returns flattened)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        return embedding

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using the model from models.py"""
        if not self.text_model:
            raise ValueError("Text model not loaded")

        # Use the model's built-in encode method
        embedding = self.text_model.encode_text(text)

        # Ensure it's 2D for FAISS (models.py returns flattened)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        return embedding

    def get_instance_metadata(self, faiss_idx: int, search_type: str) -> Dict:
        """Get metadata for a FAISS index from RDF graph"""
        if not self.rdf_graph:
            return {'faiss_index': faiss_idx}

        # Get instance URI from mapping
        if search_type == 'vision':
            instance_uri = self.vision_idx_to_instance.get(faiss_idx)
        else:
            instance_uri = self.text_idx_to_instance.get(faiss_idx)

        if not instance_uri:
            return {'faiss_index': faiss_idx}

        # Query for all properties
        query = f"""
        PREFIX smo: <http://semiomeme.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?p ?o WHERE {{
            <{instance_uri}> ?p ?o .
        }}
        """

        metadata = {
            'faiss_index': faiss_idx,
            'instance_uri': str(instance_uri)
        }

        for row in self.rdf_graph.query(query):
            predicate = str(row.p).split('/')[-1].split('#')[-1]
            value = str(row.o)

            if predicate == 'label':
                metadata['label'] = value
            elif predicate == 'kymID':
                metadata['kym_id'] = value
            elif predicate == 'hasImagePath':
                metadata['image_filename'] = value  # Just store filename
            elif predicate == 'hasOCRText':
                metadata['ocr_text'] = value
            elif predicate == 'popularity':
                metadata['popularity'] = int(value)
            elif predicate == 'belongsTo':
                metadata['parent_meme'] = value.split('/')[-1].replace('_', ' ')
            elif predicate == 'datasetType':
                metadata['dataset_type'] = value

        if self.dataset_base_dir and 'kym_id' in metadata and 'image_filename' in metadata:
            metadata['image_path'] = self.dataset_base_dir / metadata['kym_id'] / metadata['image_filename']

        return metadata

    def search_by_image(self, image_path: str, k: int = 10) -> List[Dict]:
        """Search for similar images"""
        if not self.vision_index:
            raise ValueError("Vision index not loaded")

        embedding = self.encode_image(image_path)
        distances, indices = self.vision_index.search(embedding, k)

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            metadata = self.get_instance_metadata(int(idx), 'vision')
            result = {
                'rank': i + 1,
                'index': int(idx),
                'distance': float(dist),
                'type': 'vision',
                **metadata
            }
            results.append(result)

        return results

    def search_by_text(self, text_query: str, k: int = 10) -> List[Dict]:
        """Search for similar text"""
        if not self.text_index:
            raise ValueError("Text index not loaded")

        embedding = self.encode_text(text_query)
        distances, indices = self.text_index.search(embedding, k)

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            metadata = self.get_instance_metadata(int(idx), 'text')
            result = {
                'rank': i + 1,
                'index': int(idx),
                'distance': float(dist),
                'type': 'text',
                **metadata
            }
            results.append(result)

        return results

    def search_multimodal_rrf(
            self,
            image_path: str,
            text_query: str,
            k: int = 10,
            pool_size: int = 500,
            rrf_k: int = 60
    ) -> List[Dict]:
        """
        Multiplicative RRF: Items must rank well in BOTH modalities.
        """
        vision_results = self.search_by_image(image_path, k=pool_size)
        text_results = self.search_by_text(text_query, k=pool_size)

        vision_dict = {r.get('instance_uri', str(r['index'])): r for r in vision_results}
        text_dict = {r.get('instance_uri', str(r['index'])): r for r in text_results}

        # Intersection only
        common = set(vision_dict.keys()) & set(text_dict.keys())

        print(f"   Intersection: {len(common)} items in both top-{pool_size}")

        results = []
        for instance in common:
            v = vision_dict[instance]
            t = text_dict[instance]

            # Multiplicative: must be good in both
            score = (1 / (rrf_k + v['rank'])) * (1 / (rrf_k + t['rank']))

            results.append({
                'rrf_score': score,
                'vision_rank': v['rank'],
                'text_rank': t['rank'],
                'vision_distance': v['distance'],
                'text_distance': t['distance'],
                **{k_: val for k_, val in v.items() if k_ not in ['rank', 'distance', 'type']}
            })

        results.sort(key=lambda x: x['rrf_score'], reverse=True)

        for i, r in enumerate(results[:k]):
            r['rank'] = i + 1
            r['combined_distance'] = 1 / r['rrf_score']

        return results[:k]


def create_results_grid(
        results: List[Dict],
        title: str,
        output_path: Path,
        query_image: str = None
):
    """Create a grid visualization of results"""
    n_results = min(len(results), 12)
    total_images = n_results + (1 if query_image else 0)

    cols = 4
    rows = (total_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for ax in axes:
        ax.axis('off')

    idx = 0

    # Add query image if provided
    if query_image and Path(query_image).exists():
        try:
            img = Image.open(query_image)
            axes[idx].imshow(img)
            axes[idx].set_title("QUERY", fontsize=10, fontweight='bold', color='red')
            idx += 1
        except:
            idx += 1

    # Add result images
    for result in results[:n_results]:
        if idx >= len(axes):
            break

        image_loaded = False

        # Try to load actual image
        if 'image_path' in result:
            img_path = Path(result['image_path'])

            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    axes[idx].imshow(img)
                    image_loaded = True
                except Exception as e:

                    print(f"Error loading {img_path}: {e}")

        # Fallback to placeholder
        if not image_loaded:
            placeholder = np.ones((200, 200, 3)) * 0.9
            axes[idx].imshow(placeholder)
            axes[idx].text(100, 100, f"Index\n{result['index']}",
                           ha='center', va='center', fontsize=12, color='gray')

        # Title
        dist_key = 'combined_distance' if 'combined_distance' in result else 'distance'
        title_text = f"#{result['rank']}: {result.get('parent_meme', 'Unknown')[:30]}\n"
        title_text += f"Dist: {result[dist_key]:.3f}"
        if 'kym_id' in result:
            title_text += f"\nID: {result['kym_id'][:10]}"

        axes[idx].set_title(title_text, fontsize=8)
        idx += 1

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved grid: {output_path}")




def main():
    """Main function with hardcoded paths for PyCharm execution"""

    # ============================================================
    # CONFIGURATION
    # ============================================================

    # SEARCH MODE: 'vision', 'text', 'multimodal', or 'all'
    # vision = image only, text = text only, multimodal = image+text, all = three outputs
    SEARCH_MODE = 'all'

    # Query inputs
    QUERY_IMAGE = [UPDATE PATH]
    QUERY_TEXT = "make america great"

    # Search parameters
    K = 10
    SAVE_GRIDS = True

    POOL_SIZE = 400000  # Candidates from each modality
    RRF_K = 60  # Smoothing constant (standard default)


    # Use paths from config
    VISION_INDEX_PATH = str(vision_index)
    TEXT_INDEX_PATH = str(text_index)
    VISION_MODEL_PATH = str(vision_model) if vision_model else None
    TEXT_MODEL_PATH = str(text_model) if text_model else None
    RDF_GRAPH_PATH = str(corpus_graph)
    OUTPUT_DIR = output_dir
    DATASET_BASE_DIR = dataset_base_dir

    # ============================================================
    # END CONFIGURATION
    # ============================================================

    # Initialize searcher
    searcher = SimilaritySearch(
        vision_index_path=VISION_INDEX_PATH,
        text_index_path=TEXT_INDEX_PATH,
        vision_model_path=VISION_MODEL_PATH,
        text_model_path=TEXT_MODEL_PATH,
        rdf_graph_path=RDF_GRAPH_PATH,
        dataset_base_dir=DATASET_BASE_DIR,
        search_mode=SEARCH_MODE
    )

    # Create output directory
    output_dir_path = Path(OUTPUT_DIR)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_all = {}

    print("\n" + "=" * 60)
    print("INDEX DIAGNOSTICS")
    print("=" * 60)
    print(f"Vision index vectors: {searcher.vision_index.ntotal}")
    print(f"Vision RDF mappings: {len(searcher.vision_idx_to_instance)}")
    print(f"Text index vectors: {searcher.text_index.ntotal}")
    print(f"Text RDF mappings: {len(searcher.text_idx_to_instance)}")

    print("\n" + "=" * 60)
    print("DISTANCE DISTRIBUTION ANALYSIS")
    print("=" * 60)

    v_dists, t_dists = searcher.analyse_distance_distribution(
        QUERY_IMAGE,
        QUERY_TEXT,
        k=1000
    )

    print("\n" + "=" * 60)
    print("SIMILARITY SEARCH DEMONSTRATION")
    print("=" * 60)

    # Run searches based on mode
    if SEARCH_MODE in ['vision', 'all']:
        if QUERY_IMAGE and Path(QUERY_IMAGE).exists():
            print(f"\nVISION SEARCH")
            print(f"   Query: {QUERY_IMAGE}")

            start = time.time()
            vision_results = searcher.search_by_image(QUERY_IMAGE, k=K)
            elapsed = time.time() - start

            print(f"   Time: {elapsed:.3f}s")
            print(f"   Results: {len(vision_results)}")

            for r in vision_results[:3]:
                print(f"   #{r['rank']}: Distance={r['distance']:.4f}, Parent={r.get('parent_meme', 'Unknown')}")

            results_all['vision'] = vision_results

            if SAVE_GRIDS:
                grid_path = output_dir_path / f"vision_{timestamp}.png"
                create_results_grid(vision_results, "Vision Search Results", grid_path, QUERY_IMAGE)

    if SEARCH_MODE in ['text', 'all']:
        if QUERY_TEXT:
            print(f"\nTEXT SEARCH")
            print(f"   Query: '{QUERY_TEXT}'")

            start = time.time()
            text_results = searcher.search_by_text(QUERY_TEXT, k=K)
            elapsed = time.time() - start

            print(f"   Time: {elapsed:.3f}s")
            print(f"   Results: {len(text_results)}")

            for r in text_results[:3]:
                print(f"   #{r['rank']}: Distance={r['distance']:.4f}, Parent={r.get('parent_meme', 'Unknown')}")

            results_all['text'] = text_results

            if SAVE_GRIDS:
                grid_path = output_dir_path / f"text_{timestamp}.png"
                create_results_grid(text_results, f"Text Search: '{QUERY_TEXT[:50]}'", grid_path, None)

    if SEARCH_MODE in ['multimodal', 'all']:
        if QUERY_IMAGE and Path(QUERY_IMAGE).exists() and QUERY_TEXT:
            print(f"\nMULTIMODAL SEARCH (RRF)")
            print(f"   Image: {QUERY_IMAGE}")
            print(f"   Text: '{QUERY_TEXT}'")

            start = time.time()
            multi_results = searcher.search_multimodal_rrf(
                QUERY_IMAGE, QUERY_TEXT,
                k=K,
                pool_size=POOL_SIZE,
                rrf_k=RRF_K
            )
            elapsed = time.time() - start

            print(f"   Time: {elapsed:.3f}s")
            print(f"   Results: {len(multi_results)}")

            for r in multi_results[:3]:
                print(
                    f"   #{r['rank']}: V_rank={r['vision_rank']}, T_rank={r['text_rank']}, Parent={r.get('parent_meme', 'Unknown')}")

            results_all['multimodal'] = multi_results

            if SAVE_GRIDS:
                grid_path = output_dir_path / f"multimodal_{timestamp}.png"
                create_results_grid(multi_results, f"RRF Search: '{QUERY_TEXT}'", grid_path, QUERY_IMAGE)

    # Save JSON results
    json_path = output_dir_path / f"results_{SEARCH_MODE}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'search_mode': SEARCH_MODE,
            'query': {'image': QUERY_IMAGE, 'text': QUERY_TEXT},
            'parameters': {'k': K},
            'results': results_all
        }, f, indent=2, default=str)

    print(f"\n" + "=" * 60)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
