# build_corpus_from_indices.py
"""
Refactored corpus builder that:
1. Creates unified instances
2. Stores FAISS indices directly in RDF graph
3. Still generates bridge mappings JSON for compatibility
4. Produces two graphs (combined and instance-only)
"""

import os
import sys
import json
import pickle
import faiss
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config.config import CORPUS_CONFIG

from rdflib import Graph, Literal, URIRef, RDF, RDFS, XSD

from config.namespaces import SMO, EX, SCHEMA
from config.config import CORPUS_CONFIG, Config

OUTPUT_DIR = CORPUS_CONFIG.OUTPUT_DIR
EXISTING_GRAPH_PATH = CORPUS_CONFIG.EXISTING_GRAPH_PATH
INSTANCE_GRAPH_PATH = CORPUS_CONFIG.INSTANCE_GRAPH_PATH
FAISS_INDEX_DIR = CORPUS_CONFIG.FAISS_INDEX_DIR


class IndexToCorpusBuilder:
    """Builds corpus layer using pre-built FAISS indices and metadata"""

    def __init__(self, graph_path: str, index_config: Dict):
        """
        Initialize builder with RDF graph and index configuration

        Args:
            graph_path: Path to base RDF graph with KYM entries
            index_config: Dictionary with paths to indices and mappings
                {
                    'vision_index': 'path/to/vision.faiss',
                    'vision_mappings': 'path/to/vision_mappings.pkl',
                    'text_index': 'path/to/text.faiss',
                    'text_mappings': 'path/to/text_mappings.pkl'
                }
        """
        self.graph_path = graph_path
        self.index_config = index_config
        self.bridge_mappings = {'vision': {}, 'text': {}}
        self.unified_instances = {}  # filename -> instance data

        # Load RDF graph
        print(f"Loading RDF graph from {graph_path}...")
        self.graph = Graph()
        self.graph.parse(graph_path, format="turtle")
        print(f"Loaded graph with {len(self.graph)} triples")

        # Bind namespaces
        self.graph.bind("smo", SMO)
        self.graph.bind("ex", EX)
        self.graph.bind("schema", SCHEMA)
        self.graph.bind("rdfs", RDFS)

        # Create output directories
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.faiss_output_dir = Path(FAISS_INDEX_DIR)
        self.faiss_output_dir.mkdir(parents=True, exist_ok=True)

        # Cache for KYM ID lookups
        self.kym_id_to_parent_uri = {}

    def load_indices_and_mappings(self):
        """Load pre-built FAISS indices and their mappings"""
        print("\nLoading pre-built indices and mappings...")

        # Load vision index and mappings
        if 'vision_index' in self.index_config:
            print(f"Loading vision index from {self.index_config['vision_index']}")
            self.vision_index = faiss.read_index(self.index_config['vision_index'])

            with open(self.index_config['vision_mappings'], 'rb') as f:
                self.vision_mappings = pickle.load(f)

            print(f"Vision index: {self.vision_index.ntotal} vectors")
            print(f"Vision classes: {len(set(self.vision_mappings['idx_to_class'].values()))}")

        # Load text index and mappings
        if 'text_index' in self.index_config:
            print(f"Loading text index from {self.index_config['text_index']}")
            self.text_index = faiss.read_index(self.index_config['text_index'])

            with open(self.index_config['text_mappings'], 'rb') as f:
                self.text_mappings = pickle.load(f)

            print(f"Text index: {self.text_index.ntotal} vectors")
            print(f"Text classes: {len(set(self.text_mappings['idx_to_class'].values()))}")

    def build_kym_id_cache(self):
        """Build cache of KYM ID to parent URI mappings"""
        print("\nBuilding KYM ID to parent URI cache...")

        query = """
            PREFIX smo: <http://semiomeme.org/ontology/>
            SELECT ?entity ?kymID WHERE {
                ?entity smo:kymID ?kymID .
            }
        """

        results = self.graph.query(query)
        for row in results:
            kym_id = str(row.kymID)
            parent_uri = row.entity
            self.kym_id_to_parent_uri[kym_id] = parent_uri

        print(f"Cached {len(self.kym_id_to_parent_uri)} KYM ID mappings")

    def find_parent_by_kym_id(self, kym_id: str) -> Optional[URIRef]:
        """Find parent KYM entry for given KYM ID"""
        return self.kym_id_to_parent_uri.get(kym_id)

    def generate_instance_uri(self, kym_id: str, unique_key: str) -> URIRef:
        """Generate unique URI for meme instance using a unique key"""
        # Clean KYM ID for URI
        clean_id = kym_id.replace('-', '_').replace(' ', '_').lower()

        # Use hash of unique key for instance identifier
        instance_hash = str(hash(unique_key) & 0xFFFFFF)

        # Create URI: ex:pepe_the_frog_instance_abc123
        instance_uri = URIRef(EX[f"{clean_id}_instance_{instance_hash}"])

        return instance_uri

    def create_unified_rdf_instance(self, instance_uri: URIRef, parent_uri: URIRef,
                                    vision_idx: Optional[int], text_idx: Optional[int],
                                    metadata: Dict):
        """Create a unified RDF instance with both modalities and FAISS indices"""

        # Type declaration
        self.graph.add((instance_uri, RDF.type, SMO.MemeInstance))

        # Link to parent KYM entry
        self.graph.add((instance_uri, SMO.belongsTo, parent_uri))

        # Store FAISS indices directly in RDF
        if vision_idx is not None:
            self.graph.add((instance_uri, SMO.faissVisionIndex, Literal(vision_idx, datatype=XSD.integer)))

        if text_idx is not None:
            self.graph.add((instance_uri, SMO.faissTextIndex, Literal(text_idx, datatype=XSD.integer)))

        # Add modality information
        modalities = []
        if vision_idx is not None:
            modalities.append("vision")
        if text_idx is not None:
            modalities.append("text")

        for modality in modalities:
            self.graph.add((instance_uri, SMO.hasModality, Literal(modality)))

        # Add metadata
        if 'filename' in metadata and metadata['filename']:
            self.graph.add((instance_uri, SMO.hasImagePath, Literal(metadata['filename'])))

        if 'dataset_type' in metadata and metadata['dataset_type']:
            self.graph.add((instance_uri, SMO.datasetType, Literal(metadata['dataset_type'])))

        if 'popularity' in metadata and metadata['popularity']:
            self.graph.add((instance_uri, SMO.popularity, Literal(metadata['popularity'])))

        # Add KYM ID and label
        if 'kym_id' in metadata:
            self.graph.add((instance_uri, SMO.kymID, Literal(metadata['kym_id'])))
            label = f"{metadata['kym_id']} instance"
            self.graph.add((instance_uri, RDFS.label, Literal(label)))

    def process_indices_unified(self):
        """Process both vision and text indices together to create unified instances"""
        print("\n=== Processing Unified Instances ===")

        # First pass: collect all unique instances by filename
        instance_data = {}  # filename -> data

        # Process vision index
        if hasattr(self, 'vision_index'):
            print("Processing vision embeddings...")
            for idx in tqdm(range(self.vision_index.ntotal), desc="Vision"):
                filename = self.vision_mappings.get('idx_to_filename', {}).get(idx)
                kym_id = self.vision_mappings.get('idx_to_class', {}).get(idx)

                if not filename or not kym_id:
                    continue

                if filename not in instance_data:
                    instance_data[filename] = {
                        'kym_id': kym_id,
                        'vision_idx': idx,
                        'text_idx': None,
                        'dataset_type': self.vision_mappings.get('idx_to_dataset_type', {}).get(idx, 'confirmed'),
                        'popularity': self.vision_mappings.get('idx_to_popularity', {}).get(idx)
                    }
                else:
                    instance_data[filename]['vision_idx'] = idx

        # Process text index
        if hasattr(self, 'text_index'):
            print("Processing text embeddings...")
            for idx in tqdm(range(self.text_index.ntotal), desc="Text"):
                filename = self.text_mappings.get('idx_to_filename', {}).get(idx)
                kym_id = self.text_mappings.get('idx_to_class', {}).get(idx)

                if not filename or not kym_id:
                    continue

                if filename not in instance_data:
                    instance_data[filename] = {
                        'kym_id': kym_id,
                        'vision_idx': None,
                        'text_idx': idx,
                        'dataset_type': self.text_mappings.get('idx_to_dataset_type', {}).get(idx, 'confirmed'),
                        'popularity': self.text_mappings.get('idx_to_popularity', {}).get(idx)
                    }
                else:
                    instance_data[filename]['text_idx'] = idx

        # Second pass: create RDF instances
        print(f"\nCreating RDF instances for {len(instance_data)} unique memes...")
        created = 0
        skipped_no_parent = 0

        for filename, data in tqdm(instance_data.items(), desc="Creating instances"):
            kym_id = data['kym_id']

            # Find parent
            parent_uri = self.find_parent_by_kym_id(kym_id)
            if not parent_uri:
                skipped_no_parent += 1
                continue

            # Generate instance URI using filename as unique key
            instance_uri = self.generate_instance_uri(kym_id, filename)

            # Create unified instance with all available data
            metadata = {
                'filename': filename,
                'kym_id': kym_id,
                'dataset_type': data.get('dataset_type'),
                'popularity': data.get('popularity')
            }

            self.create_unified_rdf_instance(
                instance_uri,
                parent_uri,
                data.get('vision_idx'),
                data.get('text_idx'),
                metadata
            )

            # Build bridge mappings for compatibility
            if data.get('vision_idx') is not None:
                self.bridge_mappings['vision'][str(data['vision_idx'])] = str(instance_uri)
            if data.get('text_idx') is not None:
                self.bridge_mappings['text'][str(data['text_idx'])] = str(instance_uri)

            created += 1

        print(f"\nUnified processing complete:")
        print(f"  Created: {created} unified instances")
        print(f"  Skipped (no parent): {skipped_no_parent}")

        # Statistics
        vision_only = sum(1 for d in instance_data.values() if d['vision_idx'] is not None and d['text_idx'] is None)
        text_only = sum(1 for d in instance_data.values() if d['text_idx'] is not None and d['vision_idx'] is None)
        both = sum(1 for d in instance_data.values() if d['vision_idx'] is not None and d['text_idx'] is not None)

        print(f"\nModality distribution:")
        print(f"  Vision only: {vision_only}")
        print(f"  Text only: {text_only}")
        print(f"  Both modalities: {both}")

    def save_outputs(self):
        """Save extended RDF graph, instance-only graph, and bridge mappings"""
        print("\n=== Saving Outputs ===")

        # Save extended RDF graph (combined: base + instances)
        graph_output = Path(CORPUS_CONFIG.INSTANCE_GRAPH_PATH)
        print(f"Saving combined RDF graph to {graph_output}...")
        self.graph.serialize(destination=str(graph_output), format="turtle")
        print(f"Saved combined graph with {len(self.graph)} triples")

        # Create and save instance-only graph
        print("\nCreating instance-only graph...")
        instance_graph = Graph()

        # Bind namespaces
        instance_graph.bind("smo", SMO)
        instance_graph.bind("ex", EX)
        instance_graph.bind("schema", SCHEMA)
        instance_graph.bind("rdfs", RDFS)

        # Find all MemeInstance subjects
        instance_subjects = set(self.graph.subjects(RDF.type, SMO.MemeInstance))

        # Copy all triples where subject is a MemeInstance
        instance_triple_count = 0
        for instance_uri in instance_subjects:
            for p, o in self.graph.predicate_objects(instance_uri):
                instance_graph.add((instance_uri, p, o))
                instance_triple_count += 1

        # Save instance-only graph
        instance_output = Path(CORPUS_CONFIG.INSTANCE_ONLY_GRAPH_PATH)
        print(f"Saving instance-only graph to {instance_output}...")
        instance_graph.serialize(destination=str(instance_output), format="turtle")
        print(f"Saved instance graph with {instance_triple_count} triples ({len(instance_subjects)} instances)")

        # Save bridge mappings (for compatibility)
        bridge_output = Path(CORPUS_CONFIG.BRIDGE_MAPPINGS_FILE)
        print(f"\nSaving bridge mappings to {bridge_output}...")
        with open(bridge_output, 'w') as f:
            json.dump(self.bridge_mappings, f, indent=2)
        print(f"  Vision mappings: {len(self.bridge_mappings['vision'])}")
        print(f"  Text mappings: {len(self.bridge_mappings['text'])}")

        # Save summary with detailed statistics
        summary = {
            'total_unique_instances': len(instance_subjects),
            'vision_faiss_entries': len(self.bridge_mappings['vision']),
            'text_faiss_entries': len(self.bridge_mappings['text']),
            'combined_graph_triples': len(self.graph),
            'instance_graph_triples': instance_triple_count,
            'unique_kym_entries': len(self.kym_id_to_parent_uri),
            'source_indices': self.index_config
        }

        summary_output = self.output_dir / "corpus_build_summary.json"
        with open(summary_output, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nCorpus build complete! Summary saved to {summary_output}")

    def build(self):
        """Main execution pipeline"""
        print("\n" + "=" * 50)
        print("Starting Corpus Build from Existing Indices")
        print("=" * 50)

        # Load indices
        self.load_indices_and_mappings()

        # Build KYM ID cache for fast lookups
        self.build_kym_id_cache()

        # Process indices in unified manner
        self.process_indices_unified()

        # Save everything
        self.save_outputs()

        # Print final statistics
        print("\n" + "=" * 50)
        print("Final Statistics:")
        print(f"  Unique instances created: {len(set(self.graph.subjects(RDF.type, SMO.MemeInstance)))}")
        print(f"  Vision FAISS mappings: {len(self.bridge_mappings['vision'])}")
        print(f"  Text FAISS mappings: {len(self.bridge_mappings['text'])}")
        print(f"  Total RDF triples: {len(self.graph)}")
        print("=" * 50)


if __name__ == "__main__":
    # Use pre-configured index paths
    index_config = {k: str(v) for k, v in CORPUS_CONFIG.INDEX_CONFIG.items()}

    # Build corpus
    builder = IndexToCorpusBuilder(
        graph_path=str(CORPUS_CONFIG.EXISTING_GRAPH_PATH),
        index_config=index_config
    )

    builder.build()