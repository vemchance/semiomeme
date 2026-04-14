# SemioMeme

A three-layer knowledge graph system for internet meme analysis, combining symbolic RDF representations with neural embeddings.

Accompanying repo: https://github.com/vemchance/kym_scraper (KYM WebScraper)

## Overview

SemioMeme integrates:
- **Meta Layer**: RDF knowledge graph of meme concepts from KnowYourMeme (entities, relationships, Wikidata links)
- **Corpus Layer**: Individual meme instances linked to the meta layer with FAISS index pointers
- **Retrieval Layer**: FAISS indices for vision (SigLIP) and text (Sentence-BERT) similarity search

## Requirements

```
pip install -r requirements.txt
```

Core dependencies: `rdflib`, `torch`, `transformers`, `sentence-transformers`, `faiss-cpu`, `pandas`, `numpy`

## Project Structure

```
semiomeme/
├── config/                 # Configuration and model definitions
│   ├── config.py          # Paths and settings for all layers
│   ├── models.py          # Projection head architectures
│   └── namespaces.py      # RDF namespace definitions
├── kgraph/                 # Core library
│   ├── builder/           # Graph construction (REBEL, pipelines)
│   ├── entities/          # Entity resolution
│   └── analysis/          # Ontology analysis
├── 01_Meta/               # Meta layer construction
├── 02_Retrieval/          # Embedding extraction and indexing
├── 03_Corpus/             # Corpus layer construction
├── 04_Analysis/           # Query and analysis scripts
├── data/                  # Data files (not tracked)
└── outputs/               # Generated outputs (not tracked)
```

## Setup

1. Update paths in `config/config.py`:
   - `KYM_DATA_DIR`: Path to KnowYourMeme image directories
   - Other paths are relative to project root

2. Create directories:
```python
from config.config import Config
Config.create_directories()
```

## Pipeline

### 1. Meta Layer

Build the RDF knowledge graph from KYM data:

```bash
cd 01_Meta
python 01_preprocess_data.py    # Clean raw CSV data
python 02_build_graph.py        # Build RDF graph with REBEL extraction
python 03_analyse_ontology.py   # Optional: generate statistics
```

Output: `outputs/meta/graphs/meta_graph.ttl`

### 2. Retrieval Layer

Extract embeddings and build FAISS indices:

```bash
cd 02_Retrieval
python 01_extract_image_embeddings.py   # SigLIP embeddings
python 01_extract_text_embeddings.py    # Sentence-BERT from OCR
python 02_build_image_index.py          # Build FAISS indices
python 02_build_text_index.py           # Build text indices
```

Optional finetuning (in `02_Retrieval/finetuning/`):
```bash
python finetune_siglip.py    # Train vision projection head
python finetune_text.py      # Train text projection head
```

### 3. Corpus Layer

Create meme instances and link to meta layer:

```bash
cd 03_Corpus
python 01_corpus_builder.py    # Create instances from FAISS indices
python 02_text_data.py         # Add OCR text to instances
python 03_extract_rebel.py     # Extract REBEL relations from OCR
python 04_add_rebel.py         # Add relations to graph
```

Output: `outputs/corpus/corpus_graph.ttl`

### 4. Analysis

Query scripts for similarity search and semantic queries:

```bash
cd 04_Analysis
python meta_semantic_queries.py           # SPARQL queries
python corpus_faiss_similarity_queries.py # Similarity search
python hybrid_query.py                    # Combined queries
python full_graph_statistics.py           # Generate statistics
```

## Data

Data files are available separately via Zenodo: https://zenodo.org/records/17826799
Required data structure:
```
data/
├── meta_data/
│   ├── raw/                    # Raw KYM CSV exports
│   └── cleaned_data/           # Processed data
├── retrieval_data/
│   ├── vision_embeddings/      # Image embedding chunks
│   ├── text_embeddings/        # Text embedding chunks
│   └── *_indices/              # FAISS indices
└── corpus_data/
    ├── ocr/                    # OCR CSV files
    └── faiss_indexes/          # Deployed indices
```

## Citation

Paper: [pending - accepted at ICWSM 2026] Full citation to follow upon publication

## License

Code: MIT  
Data: CC-BY-4.0
