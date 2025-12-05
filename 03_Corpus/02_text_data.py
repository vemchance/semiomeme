
"""
Post-processing script to add OCR text and REBEL-extracted relationships to existing RDF graphs.
Uses the same REBEL approach as meta layer with ontology mapping.
"""

import pandas as pd
from pathlib import Path
from rdflib import Graph, Literal, URIRef, Namespace, RDF, XSD
from tqdm import tqdm
import os
import torch
import gc
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config import CORPUS_CONFIG
from config.namespaces import SMO, EX, SCHEMA

# Settings from config
TEXT_CSV_FILES = CORPUS_CONFIG.TEXT_CSV_FILES
OUTPUT_DIR = CORPUS_CONFIG.OUTPUT_DIR
INSTANCE_GRAPH_PATH = CORPUS_CONFIG.INSTANCE_GRAPH_PATH
INSTANCE_ONLY_GRAPH_PATH = CORPUS_CONFIG.INSTANCE_ONLY_GRAPH_PATH

# Import REBEL extraction components
from kgraph.builder.rebel_extractor import REBELTripleExtractor, process_rebel_relations
from kgraph.utils import clean_uri_string, infer_entity_type_from_context
from kgraph.entities.add_entity_to_graph import add_entity_to_graph
from kgraph.entities.entity_resolver import EntityTypeResolver
from kgraph.builder.rebel_mapping_generator import load_rebel_mappings


class NoOpOntologyMapper:
    """
    Minimal ontology mapper for corpus layer that skips external enrichment.
    Used to satisfy the interface expected by add_entity_to_graph() without
    doing expensive Wikidata linking for OCR-extracted entities.
    """

    def map_entity_to_ontologies(self, graph, entity_uri, entity_type_class, namespace, label=None):
        """Do nothing - no external enrichment for corpus layer"""
        pass

    def get_ontology_class(self, entity_type):
        """Return the entity type as-is (not used in our flow but may be called)"""
        return entity_type


class TextElementEnricher:
    """Adds OCR text and REBEL-extracted relationships to existing RDF instance graphs"""

    def __init__(self, graph_paths=None, enable_rebel=False):
        """
        Initialize with paths to graphs to enrich

        Args:
            graph_paths: List of paths to RDF graphs to enrich.
                        If None, uses both default graphs from settings
            enable_rebel: Whether to enable REBEL extraction from OCR text
        """
        if graph_paths is None:
            # Use both graphs by default
            self.graph_paths = [
                str(INSTANCE_GRAPH_PATH),  # Combined graph
                str(INSTANCE_ONLY_GRAPH_PATH)  # Instance-only graph
            ]
        else:
            self.graph_paths = graph_paths

        self.text_data = {}
        self.enable_rebel = enable_rebel
        self.rebel_extractor = None
        self.entity_resolver = None
        self.ontology_mapper = None
        self.rebel_mappings = None

        self.stats = {
            'total_instances': 0,
            'instances_with_ocr': 0,
            'instances_without_ocr': 0,
            'ocr_added': 0,
            'ocr_updated': 0,
            'rebel_relations_extracted': 0,
            'rebel_entities_extracted': 0,
            'instances_with_rebel': 0
        }

        # Initialize REBEL if enabled
        if self.enable_rebel:
            print("\n=== Initializing REBEL Extractor ===")
            try:
                # Initialize REBEL extractor (meta layer - same as descriptions)
                self.rebel_extractor = REBELTripleExtractor(
                    model_name="Babelscape/rebel-large"
                )

                # Set batch size for GPU processing
                self.rebel_extractor.default_batch_size = 128  # Adjust based on GPU memory
                print(f"REBEL GPU batch size set to: {self.rebel_extractor.default_batch_size}")

                # Initialize entity processing tools (same as meta layer)
                self.entity_resolver = EntityTypeResolver()
                self.ontology_mapper = NoOpOntologyMapper()  # Stub - no Wikidata linking

                # Try to load canonical REBEL mappings
                print("Loading canonical REBEL mappings...")
                self.rebel_mappings = load_rebel_mappings(cache_dir="cache")
                if self.rebel_mappings:
                    print(f"  Loaded {len(self.rebel_mappings)} canonical mappings")
                else:
                    print("  No mappings found - using raw REBEL relations")

                print("REBEL extractor initialized successfully")

            except Exception as e:
                print(f"Warning: Failed to initialize REBEL: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing without REBEL extraction")
                self.enable_rebel = False

    def load_ocr_data(self):
        """Load OCR text from CSV files"""
        print("\n=== Loading OCR Data ===")

        for csv_file in TEXT_CSV_FILES:
            if not Path(csv_file).exists():
                print(f"Warning: CSV file not found: {csv_file}")
                continue

            try:
                print(f"Loading: {os.path.basename(csv_file)}")
                df = pd.read_csv(csv_file)

                # Check which column contains filenames
                if 'Image Ref' in df.columns:
                    filename_column = 'Image Ref'
                elif 'file' in df.columns:
                    filename_column = 'file'
                else:
                    print(f"  Warning: No filename column found. Columns: {list(df.columns)}")
                    continue

                # Check for text column
                if 'Text' not in df.columns:
                    print(f"  Warning: No 'Text' column found")
                    continue

                # Process each row
                loaded_count = 0
                for _, row in df.iterrows():
                    # Extract just the filename (not full path)
                    full_path = row[filename_column]
                    filename = full_path.split('\\')[-1] if '\\' in full_path else os.path.basename(full_path)

                    # Store text if not empty
                    text = row['Text']
                    if pd.notna(text) and str(text).strip():
                        self.text_data[filename] = str(text).strip()
                        loaded_count += 1

                print(f"  Loaded {loaded_count} text entries")

            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")

        print(f"\nTotal OCR entries loaded: {len(self.text_data)}")

        # Show sample entries
        if self.text_data:
            print("\nSample OCR entries:")
            for filename, text in list(self.text_data.items())[:3]:
                text_preview = text[:100] + "..." if len(text) > 100 else text
                print(f"  {filename}: {text_preview}")

    def add_ocr_only(self, graph_path: str):
        """Add OCR text only and save - separate from REBEL extraction"""
        print(f"\n=== Processing Graph (OCR ONLY): {Path(graph_path).name} ===")

        if not Path(graph_path).exists():
            print(f"Graph file not found: {graph_path}")
            return

        # Load graph
        print("Loading graph...")
        graph = Graph()
        graph.parse(graph_path, format="turtle")
        initial_triples = len(graph)
        print(f"Loaded graph with {initial_triples} triples")

        # Query for all MemeInstance objects
        query = """
            PREFIX smo: <http://semiomeme.org/ontology/>
            SELECT ?instance ?imagePath WHERE {
                ?instance a smo:MemeInstance ;
                         smo:hasImagePath ?imagePath .
            }
        """

        results = graph.query(query)
        instances = list(results)
        print(f"Found {len(instances)} MemeInstance objects")

        # Process each instance
        added_count = 0
        updated_count = 0
        not_found_count = 0

        for instance, image_path in tqdm(instances, desc="Adding OCR text"):
            self.stats['total_instances'] += 1

            # Get filename from image path
            filename = str(image_path)

            # Check if we have OCR text for this file
            if filename in self.text_data:
                ocr_text = self.text_data[filename]

                # Check if OCR already exists
                existing_ocr = list(graph.objects(instance, SMO.hasOCRText))

                if existing_ocr:
                    # Update existing OCR if different
                    if str(existing_ocr[0]) != ocr_text:
                        graph.remove((instance, SMO.hasOCRText, existing_ocr[0]))
                        graph.add((instance, SMO.hasOCRText, Literal(ocr_text)))
                        updated_count += 1
                        self.stats['ocr_updated'] += 1
                else:
                    # Add new OCR
                    graph.add((instance, SMO.hasOCRText, Literal(ocr_text)))
                    added_count += 1
                    self.stats['ocr_added'] += 1

                self.stats['instances_with_ocr'] += 1
            else:
                not_found_count += 1
                self.stats['instances_without_ocr'] += 1

        # Save graph with OCR
        if added_count > 0 or updated_count > 0:
            print(f"\nSaving graph with OCR...")
            graph.serialize(destination=graph_path, format="turtle")

            final_triples = len(graph)
            print(f"Graph saved:")
            print(f"  Initial triples: {initial_triples}")
            print(f"  Final triples: {final_triples}")
            print(f"  OCR added: {added_count}")
            print(f"  OCR updated: {updated_count}")
            print(f"  No OCR found: {not_found_count}")
        else:
            print("No OCR changes made")

    def add_rebel_only(self, graph_path: str):
        """Add REBEL relations only - assumes OCR already exists"""
        print(f"\n=== Processing Graph (REBEL ONLY): {Path(graph_path).name} ===")

        if not self.enable_rebel or not self.rebel_extractor:
            print("REBEL extraction not enabled")
            return

        if not Path(graph_path).exists():
            print(f"Graph file not found: {graph_path}")
            return

        # Load graph
        print("Loading graph...")
        graph = Graph()
        graph.parse(graph_path, format="turtle")
        initial_triples = len(graph)
        print(f"Loaded graph with {initial_triples} triples")

        # Query for instances WITH OCR text
        query = """
            PREFIX smo: <http://semiomeme.org/ontology/>
            SELECT ?instance ?ocrText WHERE {
                ?instance a smo:MemeInstance ;
                         smo:hasOCRText ?ocrText .
            }
        """

        results = graph.query(query)
        instances_with_ocr = []
        ocr_texts = []

        for instance, ocr_text in results:
            instances_with_ocr.append(instance)
            ocr_texts.append(str(ocr_text))

        print(f"Found {len(instances_with_ocr)} instances with OCR text")

        if not ocr_texts:
            print("No OCR text found - run add_ocr_only() first")
            return

        # Process REBEL extraction
        print(f"\n=== Extracting REBEL Relations ===")

        if self.rebel_mappings:
            print(f"Using {len(self.rebel_mappings)} canonical relation mappings")

        # Track unique entities
        unique_entities = set()

        # Create entity registry for this graph
        entity_registry = {}
        entity_types = set()

        # Pre-populate registry from existing entities in graph to avoid duplicates
        print("Pre-populating entity registry from existing graph...")
        existing_count = 0
        for entity in graph.subjects(RDF.type, None):
            entity_uri_str = str(entity).split('/')[-1]
            entity_type_list = list(graph.objects(entity, RDF.type))
            if entity_type_list:
                entity_type = entity_type_list[0]
                entity_registry[entity_uri_str] = {'uri': entity, 'type': entity_type}
                entity_types.add(entity_type)
                existing_count += 1

        print(f"  Found {existing_count} existing entities in graph")
        print(f"  {len(entity_types)} unique entity types")

        # Process in chunks for better progress visibility
        chunk_size = 5000  # Process 5000 texts at a time for progress updates
        total_chunks = (len(ocr_texts) + chunk_size - 1) // chunk_size

        print(f"\nProcessing {len(ocr_texts)} texts in {total_chunks} chunks of {chunk_size}...")

        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(ocr_texts))

            chunk_texts = ocr_texts[chunk_start:chunk_end]
            chunk_instances = instances_with_ocr[chunk_start:chunk_end]

            print(f"\n=== Chunk {chunk_idx + 1}/{total_chunks}: Processing texts {chunk_start}-{chunk_end} ===")

            # Extract relations using META LAYER extractor (direct batch processing)
            batch_relations = self.rebel_extractor.extract_triples_batch(
                chunk_texts,
                batch_size=None,  # Uses default_batch_size
                show_progress=True
            )

            # Add relations to graph
            print(f"Adding relations to graph for chunk {chunk_idx + 1}...")
            for instance_node, relations in zip(chunk_instances, batch_relations):
                if relations:
                    # Count unique entities
                    for subject, relation, obj in relations:
                        unique_entities.add(subject.strip())
                        unique_entities.add(obj.strip())

                    # Add relations using meta layer function
                    num_added = process_rebel_relations(
                        g=graph,
                        rebel_relations=relations,
                        main_entity_node=instance_node,
                        smo_namespace=SMO,
                        ex_namespace=EX,
                        entity_registry=entity_registry,
                        entity_types=entity_types,
                        entity_resolver=self.entity_resolver,
                        ontology_mapper=self.ontology_mapper,
                        rebel_mappings=self.rebel_mappings
                    )

                    # Add source provenance
                    if num_added > 0:
                        for pred in graph.subjects(SMO.extractedBy, Literal("REBEL")):
                            if not list(graph.objects(pred, SMO.extractedFromSource)):
                                graph.add((pred, SMO.extractedFromSource, Literal("ocr_caption")))

                    if num_added > 0:
                        self.stats['rebel_relations_extracted'] += num_added
                        self.stats['instances_with_rebel'] += 1

            print(f"Chunk {chunk_idx + 1} complete: {self.stats['rebel_relations_extracted']} total relations so far")

            # Clean up GPU memory between chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        # Store unique entities
        self.stats['rebel_entities_extracted'] = len(unique_entities)

        # Save graph with REBEL
        if self.stats['rebel_relations_extracted'] > 0:
            print(f"\nSaving graph with REBEL relations...")
            graph.serialize(destination=graph_path, format="turtle")

            final_triples = len(graph)
            print(f"Graph saved:")
            print(f"  Initial triples: {initial_triples}")
            print(f"  Final triples: {final_triples}")
            print(f"  REBEL relations: {self.stats['rebel_relations_extracted']}")
            print(f"  REBEL entities: {self.stats['rebel_entities_extracted']}")
            print(f"  Instances with relations: {self.stats['instances_with_rebel']}")
        else:
            print("No REBEL relations extracted")

    def enrich_graph(self, graph_path: str):
        """Add OCR text and optionally REBEL relations to a single graph - DEPRECATED

        Use add_ocr_only() then add_rebel_only() instead for better control
        """
        print(f"\n=== Processing Graph: {Path(graph_path).name} ===")

        if not Path(graph_path).exists():
            print(f"Graph file not found: {graph_path}")
            return

        # Load graph
        print("Loading graph...")
        graph = Graph()
        graph.parse(graph_path, format="turtle")
        initial_triples = len(graph)
        print(f"Loaded graph with {initial_triples} triples")

        # Query for all MemeInstance objects
        query = """
            PREFIX smo: <http://semiomeme.org/ontology/>
            SELECT ?instance ?imagePath WHERE {
                ?instance a smo:MemeInstance ;
                         smo:hasImagePath ?imagePath .
            }
        """

        results = graph.query(query)
        instances = list(results)
        print(f"Found {len(instances)} MemeInstance objects")

        # Process each instance
        added_count = 0
        updated_count = 0
        not_found_count = 0

        # Collect OCR texts for REBEL batch processing
        instances_with_ocr = []
        ocr_texts = []

        for instance, image_path in tqdm(instances, desc="Adding OCR text"):
            self.stats['total_instances'] += 1

            # Get filename from image path
            filename = str(image_path)

            # Check if we have OCR text for this file
            if filename in self.text_data:
                ocr_text = self.text_data[filename]

                # Check if OCR already exists
                existing_ocr = list(graph.objects(instance, SMO.hasOCRText))

                if existing_ocr:
                    # Update existing OCR if different
                    if str(existing_ocr[0]) != ocr_text:
                        graph.remove((instance, SMO.hasOCRText, existing_ocr[0]))
                        graph.add((instance, SMO.hasOCRText, Literal(ocr_text)))
                        updated_count += 1
                        self.stats['ocr_updated'] += 1
                else:
                    # Add new OCR
                    graph.add((instance, SMO.hasOCRText, Literal(ocr_text)))
                    added_count += 1
                    self.stats['ocr_added'] += 1

                self.stats['instances_with_ocr'] += 1

                # Collect for REBEL processing
                instances_with_ocr.append(instance)
                ocr_texts.append(ocr_text)

            else:
                not_found_count += 1
                self.stats['instances_without_ocr'] += 1

        # Process REBEL extraction if enabled and we have OCR texts
        if self.enable_rebel and self.rebel_extractor and ocr_texts:
            print(f"\n=== Extracting REBEL Relations ===")
            print(f"Processing {len(ocr_texts)} OCR texts...")

            if self.rebel_mappings:
                print(f"Using {len(self.rebel_mappings)} canonical relation mappings")

            # Track unique entities
            unique_entities = set()

            # PRE-POPULATE entity registry from existing graph
            print(f"\nPre-populating entity registry from existing graph...")
            entity_registry = {}
            entity_types = set()

            # Query all existing entities with labels and types
            registry_query = """
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                SELECT ?entity ?label ?type WHERE {
                    ?entity rdfs:label ?label ;
                            rdf:type ?type .
                }
            """

            for row in graph.query(registry_query):
                # Extract URI part after namespace
                entity_uri = str(row.entity).split('/')[-1]

                # Store in registry
                entity_registry[entity_uri] = {
                    'uri': row.entity,
                    'type': row.type,
                    'label': str(row.label)
                }
                entity_types.add(row.type)

            print(f"Pre-populated registry with {len(entity_registry)} existing entities")
            print(f"Found {len(entity_types)} entity types")

            # Show sample of pre-populated entities
            if entity_registry:
                print("\nSample pre-populated entities:")
                sample_count = 0
                for uri, data in entity_registry.items():
                    if sample_count >= 5:
                        break
                    type_name = str(data['type']).split('/')[-1]
                    print(f"  {uri} ({type_name}): {data['label']}")
                    sample_count += 1

            print(f"\nExtracting relations from all texts...")

            # Extract relations using CORPUS extractor (fast, no chunking)
            # The extractor handles its own internal batching for GPU
            batch_relations = self.rebel_extractor.process_batch(
                ocr_texts,
                show_progress=True
            )

            # Add relations to graph using META LAYER function
            for instance_node, relations in zip(instances_with_ocr, batch_relations):
                if relations:
                    # Check if this instance already has REBEL relations (skip if re-running)
                    existing_rebel = list(graph.objects(instance_node, SMO.mentions))
                    if existing_rebel:
                        # Check if any of these mentions came from REBEL
                        has_rebel = False
                        for mention in existing_rebel:
                            # Check if there are predicates from this instance marked as REBEL
                            for pred in graph.predicates(instance_node, mention):
                                if list(graph.objects(pred, SMO.extractedBy)):
                                    has_rebel = True
                                    break
                            if has_rebel:
                                break

                        if has_rebel:
                            print(f"  Skipping instance (already has REBEL relations)")
                            continue

                    # Count unique entities from this instance's relations
                    for subject, relation, obj in relations:
                        unique_entities.add(subject.strip())
                        unique_entities.add(obj.strip())

                    # Use the EXACT SAME function as meta layer
                    # Note: We manually add source provenance since rebel_extractor.py
                    # doesn't have text_source parameter yet
                    num_added = process_rebel_relations(
                        g=graph,
                        rebel_relations=relations,
                        main_entity_node=instance_node,
                        smo_namespace=SMO,
                        ex_namespace=EX,
                        entity_registry=entity_registry,
                        entity_types=entity_types,
                        entity_resolver=self.entity_resolver,
                        ontology_mapper=self.ontology_mapper,
                        rebel_mappings=self.rebel_mappings  # Use canonical mappings
                    )

                    # Add source provenance for OCR extraction
                    # Find all predicates marked with extractedBy REBEL and add source
                    if num_added > 0:
                        for pred in graph.subjects(SMO.extractedBy, Literal("REBEL")):
                            # Only add if not already present
                            if not list(graph.objects(pred, SMO.extractedFromSource)):
                                graph.add((pred, SMO.extractedFromSource, Literal("ocr_caption")))

                    if num_added > 0:
                        self.stats['rebel_relations_extracted'] += num_added
                        self.stats['instances_with_rebel'] += 1

            # Store total unique entities
            self.stats['rebel_entities_extracted'] = len(unique_entities)

            print(f"\nREBEL extraction complete:")
            print(f"  Total relations extracted: {self.stats['rebel_relations_extracted']}")
            print(f"  Unique entities extracted: {self.stats['rebel_entities_extracted']}")
            print(f"  Instances with relations: {self.stats['instances_with_rebel']}")
            print(f"  Source provenance: 'ocr_caption' added to all predicates")

        # Save enriched graph
        if added_count > 0 or updated_count > 0 or self.stats['rebel_relations_extracted'] > 0:
            print(f"\nSaving enriched graph...")
            graph.serialize(destination=graph_path, format="turtle")

            final_triples = len(graph)
            print(f"Graph enriched:")
            print(f"  Initial triples: {initial_triples}")
            print(f"  Final triples: {final_triples}")
            print(f"  OCR added: {added_count}")
            print(f"  OCR updated: {updated_count}")
            print(f"  No OCR found: {not_found_count}")
            if self.enable_rebel:
                print(f"  REBEL relations: {self.stats['rebel_relations_extracted']}")
                print(f"  REBEL entities: {self.stats['rebel_entities_extracted']}")
        else:
            print("No changes made to graph")

    def add_visual_elements(self, graph_path: str, visual_data: dict):
        """
        Placeholder for future visual element additions

        Args:
            graph_path: Path to graph to enrich
            visual_data: Dictionary mapping filenames to visual features
                        e.g., {'file.jpg': {'objects': ['cat', 'dog'], 'colors': ['red', 'blue']}}
        """
        print("\n=== Adding Visual Elements (Future Feature) ===")
        print("This method will add visual features like:")
        print("  - Detected objects (smo:hasDetectedObject)")
        print("  - Dominant colors (smo:hasDominantColor)")
        print("  - Face count (smo:hasFaceCount)")
        print("  - Image captions (smo:hasCaption)")
        pass

    def run(self, ocr_only=False, rebel_only=False):
        """Main execution pipeline

        Args:
            ocr_only: If True, only add OCR text and save
            rebel_only: If True, only add REBEL relations (assumes OCR exists)
        """
        print("\n" + "=" * 50)
        print("Text Element Enrichment Pipeline")
        print("Using META LAYER REBEL approach")

        if ocr_only:
            print("Mode: OCR ONLY")
        elif rebel_only:
            print("Mode: REBEL ONLY")
        else:
            print("Mode: OCR + REBEL (sequential)")

        print("=" * 50)

        # Load OCR data if needed
        if not rebel_only:
            self.load_ocr_data()
            if not self.text_data:
                print("\nNo OCR data loaded. Exiting.")
                return

        # Process each graph
        for graph_path in self.graph_paths:
            if ocr_only:
                # Only add OCR and save
                self.add_ocr_only(graph_path)
            elif rebel_only:
                # Only add REBEL (OCR must exist)
                self.add_rebel_only(graph_path)
            else:
                # Do both: OCR first (save), then REBEL (save)
                self.add_ocr_only(graph_path)
                if self.enable_rebel:
                    self.add_rebel_only(graph_path)

        # Print final statistics
        self.print_statistics()

    def print_statistics(self):
        """Print enrichment statistics"""
        print("\n" + "=" * 50)
        print("Enrichment Statistics")
        print("=" * 50)
        print(f"Total instances processed: {self.stats['total_instances']}")
        print(f"Instances with OCR: {self.stats['instances_with_ocr']}")
        print(f"Instances without OCR: {self.stats['instances_without_ocr']}")
        print(f"OCR text added: {self.stats['ocr_added']}")
        print(f"OCR text updated: {self.stats['ocr_updated']}")

        if self.enable_rebel:
            print(f"\nREBEL Extraction:")
            print(f"  Relations extracted: {self.stats['rebel_relations_extracted']}")
            print(f"  Unique entities extracted: {self.stats['rebel_entities_extracted']}")
            print(f"  Instances with relations: {self.stats['instances_with_rebel']}")

        if self.stats['total_instances'] > 0:
            coverage = (self.stats['instances_with_ocr'] / self.stats['total_instances']) * 100
            print(f"\nOCR coverage: {coverage:.1f}%")

            if self.enable_rebel and self.stats['instances_with_ocr'] > 0:
                rebel_coverage = (self.stats['instances_with_rebel'] / self.stats['instances_with_ocr']) * 100
                print(f"REBEL coverage (of OCR texts): {rebel_coverage:.1f}%")


if __name__ == "__main__":
    # Initialize enricher
    enricher = TextElementEnricher(enable_rebel=True)

    # Adjust GPU batch size for meta layer extractor
    if enricher.rebel_extractor:
        enricher.rebel_extractor.default_batch_size = 128  # Increase as GPU allows
        print(f"GPU batch size set to: {enricher.rebel_extractor.default_batch_size}")

    # === OPTION 1: Add OCR only (safe to re-run) ===
    enricher.run(ocr_only=True)

    # === OPTION 2: Add REBEL only (run AFTER OCR is done) ===
    # enricher.run(rebel_only=True)

    # # === OPTION 3: Do both sequentially (OCR save, then REBEL save) ===
    # enricher.run()  # Default: does OCR first, saves, then REBEL, saves again