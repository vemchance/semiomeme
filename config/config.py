from pathlib import Path
import os
from typing import Dict, List, Optional


class Config:
    """Base configuration class with path resolution"""

    PROJECT_ROOT = Path(__file__).parent.parent

    # ==================== DIRECTORY STRUCTURE ====================

    # Main data directories
    DATA_ROOT = PROJECT_ROOT / 'data'
    OUTPUTS_ROOT = PROJECT_ROOT / 'outputs'

    # Layer-specific data directories
    META_DATA_DIR = DATA_ROOT / 'meta_data'
    RETRIEVAL_DATA_DIR = DATA_ROOT / 'retrieval_data'
    CORPUS_DATA_DIR = DATA_ROOT / 'corpus_data'

    # Layer-specific output directories
    META_OUTPUT_DIR = OUTPUTS_ROOT / 'meta'
    RETRIEVAL_OUTPUT_DIR = OUTPUTS_ROOT / 'retrieval'
    CORPUS_OUTPUT_DIR = OUTPUTS_ROOT / 'corpus'

    # External dataset paths (use environment variables for portability)
    KYM_DATA_DIR = Path('Datasets/Memes/KYM') # UPDATE TO YOUR PATH

    # ==================== SHARED SETTINGS ====================

    # Common processing settings
    BATCH_SIZE = 64  # Default batch size for all operations
    TEST_MODE = False
    SAMPLE_SIZE = 100  # For testing

    # Common model settings
    DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

    # ==================== META LAYER SETTINGS ====================

class Meta:
    # File paths
    INPUT_FILE = Config.META_DATA_DIR / 'cleaned_data' / 'cleaned_source_data.csv'
    OUTPUT_DIR = Config.META_OUTPUT_DIR / 'graphs'
    CACHE_DIR = Config.META_DATA_DIR / 'cache'
    WIKIDATA_CACHE = CACHE_DIR / 'wikidata_local'
    DEFAULT_GRAPH = OUTPUT_DIR / 'meta_graph.ttl'

    # Graph building settings
    GRAPH_BUILD = {
        'batch_size': 16,
        'test_mode': Config.TEST_MODE,
        'sample_size': Config.SAMPLE_SIZE,
        'enable_rebel': True,
        'use_local_wikidata': True,
        'enrich_rebel': False,
    }

    # Entity linking settings
    ENTITY_LINKING = {
        'rate_limit_interval': 1.0,
        'timeout': 8,
    }

    # REBEL model settings
    REBEL = {
        'model_name': 'Babelscape/rebel-large',
        'max_seq_length': 768,
        'chunk_overlap': 128,
        'beam_width': 3,
        'min_frequency': 3,
        'similarity_threshold': 0.7,
    }

    # Analysis settings
    ANALYSIS = {
        'output_dir': Config.OUTPUTS_ROOT / 'analysis',
        'use_timestamp': True,
        'visualisation_dpi': 300,
        'max_nodes': 50,
        'generate_statistics': True,
        'analyse_connectivity': True,
        'cleanup_old_sessions': True,
        'keep_recent_sessions': 3,
    }

# ==================== RETRIEVAL LAYER SETTINGS ====================

class Retrieval:
    # Data sources
    CONFIRMED_DIRS = [Config.KYM_DATA_DIR / 'Confirmed Images']
    UNCONFIRMED_DIRS = [Config.KYM_DATA_DIR / 'Unconfirmed Images']

    # Model settings
    IMAGE_MODEL_NAME = "google/siglip-base-patch16-384"
    IMAGE_EMBEDDING_DIM = 768
    TEXT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    TEXT_EMBEDDING_DIM = 768

    # Processing settings
    IMAGE_BATCH_SIZE = 64
    TEXT_BATCH_SIZE = 256
    IMAGE_CHUNK_SIZE = 50000
    TEXT_CHUNK_SIZE = 100000

    # Output directories
    OUTPUT_DIR = Config.RETRIEVAL_OUTPUT_DIR
    DATA_DIR = Config.RETRIEVAL_DATA_DIR

    EMBEDDINGS_DIR = DATA_DIR / 'vision_embeddings'
    METADATA_DIR = DATA_DIR / 'vision_metadata'
    INDEX_DIR = DATA_DIR / 'vision_indices'  # Indices stay with data
    TEXT_EMBEDDINGS_DIR = DATA_DIR / 'text_embeddings'
    TEXT_METADATA_DIR = DATA_DIR / 'text_metadata'
    TEXT_INDEX_DIR = DATA_DIR / 'text_indices'

    # OCR files
    OCR_FILES = [
        Config.CORPUS_DATA_DIR / 'ocr' / 'confirmed_memes_full.csv',
        Config.CORPUS_DATA_DIR / 'ocr' / 'unconfirmed_memes_full.csv',
    ]

    # FAISS settings
    FAISS = {
        'nlist': 2048,  # Number of clusters for IVF
        'm': 64,  # Number of subquantizers for PQ
        'bits': 8,  # Bits per subquantizer
    }

    FINETUNED_MODELS_DIR = OUTPUT_DIR / 'finetuned_models'
    VISION_FINETUNED_DIR = FINETUNED_MODELS_DIR / 'vision'
    TEXT_FINETUNED_DIR = FINETUNED_MODELS_DIR / 'text'

    VISION_BEST_MODEL = VISION_FINETUNED_DIR / 'vision_model.pth'
    TEXT_BEST_MODEL = TEXT_FINETUNED_DIR / 'text_model.pth'

    MODEL_SEARCH_PATHS = [
        FINETUNED_MODELS_DIR / 'finetuned_projection_confirmed',
        Config.OUTPUTS_ROOT / 'finetuned_projection_confirmed',
    ]

    # ==================== CORPUS LAYER SETTINGS ====================

class Corpus:
    # Input directories (reuse from Retrieval for consistency)
    IMAGE_DIRECTORIES = Retrieval.CONFIRMED_DIRS + Retrieval.UNCONFIRMED_DIRS
    TEXT_CSV_FILES = Retrieval.OCR_FILES

    # Output directories
    OUTPUT_DIR = Config.CORPUS_OUTPUT_DIR
    FAISS_INDEX_DIR = OUTPUT_DIR / 'faiss_indices'
    BRIDGE_MAPPINGS_FILE = OUTPUT_DIR / 'bridge_mappings.json'

    # Model paths - where corpus layer expects to find deployed models
    VISION_MODEL_PATH = Config.CORPUS_DATA_DIR / 'vision_model.pth'
    TEXT_MODEL_PATH = Config.CORPUS_DATA_DIR / 'text_model.pth'

    # Search paths - look in multiple locations
    VISION_MODEL_SEARCH_PATHS = [
        Config.CORPUS_DATA_DIR / 'vision_model.pth',
        Retrieval.VISION_BEST_MODEL,  # Fallback: directly from retrieval
    ]

    TEXT_MODEL_SEARCH_PATHS = [
        Config.CORPUS_DATA_DIR / 'text_model.pth',  # Primary: deployed/copied model
        Retrieval.TEXT_BEST_MODEL,  # Fallback: directly from retrieval
    ]

    # FAISS index paths (in corpus_data, not outputs)
    FAISS_DATA_DIR = Config.CORPUS_DATA_DIR / 'faiss_indexes'
    VISION_INDEX_PATH = FAISS_DATA_DIR / 'vision_index.faiss'
    VISION_MAPPINGS_PATH = FAISS_DATA_DIR / 'vision_metadata.pkl'
    TEXT_INDEX_PATH = FAISS_DATA_DIR / 'text_index.faiss'
    TEXT_MAPPINGS_PATH = FAISS_DATA_DIR / 'text_metadata.pkl'

    # Index configuration dictionary for convenience
    INDEX_CONFIG = {
        'vision_index': VISION_INDEX_PATH,
        'vision_mappings': VISION_MAPPINGS_PATH,
        'text_index': TEXT_INDEX_PATH,
        'text_mappings': TEXT_MAPPINGS_PATH
    }

    # Processing settings
    BATCH_SIZE = 1000
    SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}

    # Graph paths
    EXISTING_GRAPH_PATH = Meta.DEFAULT_GRAPH
    INSTANCE_GRAPH_PATH = OUTPUT_DIR / 'corpus_graph.ttl'
    INSTANCE_ONLY_GRAPH_PATH = OUTPUT_DIR / 'instance_graph.ttl'

@classmethod
def create_directories(cls):
    """Create all necessary directories if they don't exist"""
    directories = [
        # Main directories
        cls.DATA_ROOT,
        cls.OUTPUTS_ROOT,

        # Meta directories
        cls.Meta.CACHE_DIR,
        cls.Meta.WIKIDATA_CACHE,
        cls.Meta.OUTPUT_DIR,
        cls.META_DATA_DIR / 'cleaned_data',
        cls.META_DATA_DIR / 'raw',

        # Retrieval directories
        cls.Retrieval.EMBEDDINGS_DIR,
        cls.Retrieval.METADATA_DIR,
        cls.Retrieval.INDEX_DIR,
        cls.Retrieval.TEXT_EMBEDDINGS_DIR,
        cls.Retrieval.TEXT_METADATA_DIR,
        cls.Retrieval.TEXT_INDEX_DIR,

        # Add finetuning output directories
        cls.Retrieval.VISION_FINETUNED_DIR,
        cls.Retrieval.TEXT_FINETUNED_DIR,

        # Corpus directories
        cls.Corpus.OUTPUT_DIR,
        cls.Corpus.FAISS_INDEX_DIR,
        cls.Corpus.FAISS_DATA_DIR,
        cls.CORPUS_DATA_DIR / 'ocr',
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


@classmethod
def get_best_model_path(cls, model_type='vision'):
    """
    Get the best model path, checking multiple locations

    Args:
        model_type: 'vision' or 'text'

    Returns:
        Path to the best model if found, None otherwise
    """
    if model_type == 'vision':
        search_paths = cls.Corpus.VISION_MODEL_SEARCH_PATHS
    elif model_type == 'text':
        search_paths = cls.Corpus.TEXT_MODEL_SEARCH_PATHS
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    for path in search_paths:
        if path.exists():
            return path

    return None

@classmethod
def get_config_dict(cls, layer: Optional[str] = None) -> Dict:
    """Get configuration as dictionary, optionally for a specific layer"""
    if layer == 'meta':
        return {
            'paths': {
                'input_file': str(cls.Meta.INPUT_FILE),
                'output_dir': str(cls.Meta.OUTPUT_DIR),
                'cache_dir': str(cls.Meta.CACHE_DIR),
                'wikidata_cache': str(cls.Meta.WIKIDATA_CACHE),
                'default_graph': str(cls.Meta.DEFAULT_GRAPH),
            },
            **cls.Meta.GRAPH_BUILD,
            'entity_linking': cls.Meta.ENTITY_LINKING,
            'rebel': cls.Meta.REBEL,
            'analysis': cls.Meta.ANALYSIS,
        }
    elif layer == 'retrieval':
        return {
            'confirmed_dirs': [str(d) for d in cls.Retrieval.CONFIRMED_DIRS],
            'unconfirmed_dirs': [str(d) for d in cls.Retrieval.UNCONFIRMED_DIRS],
            'models': {
                'image': cls.Retrieval.IMAGE_MODEL_NAME,
                'text': cls.Retrieval.TEXT_MODEL_NAME,
            },
            'batch_sizes': {
                'image': cls.Retrieval.IMAGE_BATCH_SIZE,
                'text': cls.Retrieval.TEXT_BATCH_SIZE,
            },
            'output_paths': {
                'embeddings': str(cls.Retrieval.EMBEDDINGS_DIR),
                'metadata': str(cls.Retrieval.METADATA_DIR),
                'indices': str(cls.Retrieval.INDEX_DIR),
            },
            'faiss': cls.Retrieval.FAISS,
        }
    elif layer == 'corpus':
        return {
            'image_directories': [str(d) for d in cls.Corpus.IMAGE_DIRECTORIES],
            'text_csv_files': [str(f) for f in cls.Corpus.TEXT_CSV_FILES],
            'output_dir': str(cls.Corpus.OUTPUT_DIR),
            'batch_size': cls.Corpus.BATCH_SIZE,
            'graph_paths': {
                'existing': str(cls.Corpus.EXISTING_GRAPH_PATH),
                'instance': str(cls.Corpus.INSTANCE_GRAPH_PATH),
                'instance_only': str(cls.Corpus.INSTANCE_ONLY_GRAPH_PATH),
            },
        }
    else:
        # Return all config
        return {
            'project_root': str(cls.PROJECT_ROOT),
            'meta': cls.get_config_dict('meta'),
            'retrieval': cls.get_config_dict('retrieval'),
            'corpus': cls.get_config_dict('corpus'),
        }

# Create a default instance for backward compatibility
PROJECT_ROOT = Config.PROJECT_ROOT
META_CONFIG = Meta  # Direct reference to the Meta class
RETRIEVAL_CONFIG = Retrieval  # Direct reference to the Retrieval class
CORPUS_CONFIG = Corpus  # Direct reference to the Corpus class


