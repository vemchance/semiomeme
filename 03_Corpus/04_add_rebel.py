"""
Phase 2: Add extracted REBEL relations to graph with entity registry
Loads pre-extracted relations from JSON and adds them to graph with proper entity resolution
"""

import json
from pathlib import Path
from rdflib import Graph, Literal, URIRef
from tqdm import tqdm

from config.config import CORPUS_CONFIG
from config.namespaces import SMO, EX

INSTANCE_GRAPH_PATH = CORPUS_CONFIG.INSTANCE_GRAPH_PATH
INSTANCE_ONLY_GRAPH_PATH = CORPUS_CONFIG.INSTANCE_ONLY_GRAPH_PATH
OUTPUT_DIR = CORPUS_CONFIG.OUTPUT_DIR

# Import REBEL processing components
from kgraph.builder.rebel_extractor import process_rebel_relations
from kgraph.entities.entity_resolver import EntityTypeResolver
from kgraph.builder.rebel_mapping_generator import load_rebel_mappings

class NoOpOntologyMapper:
    """Stub ontology mapper - no Wikidata linking"""

    def map_entity_to_ontologies(self, graph, entity_uri, entity_type_class, namespace, label=None):
        pass

    def get_ontology_class(self, entity_type):
        return entity_type


class REBELGraphAdder:
    """Add pre-extracted REBEL relations to graph"""

    def __init__(self, extractions_file="rebel_extractions.json", graph_paths=None):
        self.extractions_file = extractions_file

        if graph_paths is None:
            self.graph_paths = [
                str(INSTANCE_GRAPH_PATH),
                str(INSTANCE_ONLY_GRAPH_PATH)
            ]
        else:
            self.graph_paths = graph_paths

        # Initialize entity processing tools
        self.entity_resolver = EntityTypeResolver()
        self.ontology_mapper = NoOpOntologyMapper()

        # Load REBEL mappings
        print("Loading canonical REBEL mappings...")
        self.rebel_mappings = load_rebel_mappings(cache_dir="cache")
        if self.rebel_mappings:
            print(f"  Loaded {len(self.rebel_mappings)} canonical mappings")
        else:
            print("  No mappings found - using raw REBEL relations")

        self.stats = {
            'rebel_relations_extracted': 0,
            'rebel_entities_extracted': 0,
            'instances_with_rebel': 0
        }

    def load_extractions(self):
        """Load pre-extracted REBEL relations from JSON"""
        print(f"\n=== Loading Extractions ===")
        print(f"Loading from: {self.extractions_file}")

        with open(self.extractions_file, 'r') as f:
            data = json.load(f)

        print(f"Total texts processed: {data['total_texts']}")
        print(f"Texts with relations: {data['texts_with_relations']}")

        # Convert back to tuples
        extractions = {}
        for filename, relations in data['extractions'].items():
            extractions[filename] = [tuple(rel) for rel in relations]

        return extractions

    def pre_populate_entity_registry(self, graph):
        """Pre-populate entity registry from existing graph"""
        print("\nPre-populating entity registry from existing graph...")
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

        # Show sample
        if entity_registry:
            print("\nSample pre-populated entities:")
            sample_count = 0
            for uri, data in entity_registry.items():
                if sample_count >= 5:
                    break
                type_name = str(data['type']).split('/')[-1]
                print(f"  {uri} ({type_name}): {data['label']}")
                sample_count += 1

        return entity_registry, entity_types

    def add_to_graph(self, graph_path: str, extractions: dict):
        """Add REBEL relations to a specific graph"""
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

        # Pre-populate entity registry
        entity_registry, entity_types = self.pre_populate_entity_registry(graph)

        # Query for all MemeInstance objects with OCR text
        print("\nQuerying for instances with OCR text...")
        query = """
            PREFIX smo: <http://semiomeme.org/ontology/>
            SELECT ?instance ?imagePath ?ocrText WHERE {
                ?instance a smo:MemeInstance ;
                         smo:hasImagePath ?imagePath ;
                         smo:hasOCRText ?ocrText .
            }
        """

        results = graph.query(query)
        instances = list(results)
        print(f"Found {len(instances)} instances with OCR text")

        # Track unique entities
        unique_entities = set()

        # Process each instance
        print("\nAdding REBEL relations to graph...")
        for instance, image_path, ocr_text in tqdm(instances, desc="Processing instances"):
            filename = str(image_path)

            # Check if we have extractions for this filename
            if filename not in extractions:
                continue

            relations = extractions[filename]
            if not relations:
                continue

            # Check if instance already has REBEL relations (skip if re-running)
            existing_rebel = list(graph.objects(instance, SMO.mentions))
            if existing_rebel:
                # Quick check if any mentions have extractedBy annotation
                has_rebel = False
                for mention in existing_rebel:
                    for pred in graph.predicates(instance, mention):
                        if list(graph.objects(pred, SMO.extractedBy)):
                            has_rebel = True
                            break
                    if has_rebel:
                        break

                if has_rebel:
                    continue

            # Count unique entities
            for subject, relation, obj in relations:
                unique_entities.add(subject.strip())
                unique_entities.add(obj.strip())

            # Add relations using meta layer function
            num_added = process_rebel_relations(
                g=graph,
                rebel_relations=relations,
                main_entity_node=instance,
                smo_namespace=SMO,
                ex_namespace=EX,
                entity_registry=entity_registry,
                entity_types=entity_types,
                entity_resolver=self.entity_resolver,
                ontology_mapper=self.ontology_mapper,
                rebel_mappings=self.rebel_mappings
            )

            # No extractedFromSource - just extractedBy "REBEL" (already added by process_rebel_relations)

            if num_added > 0:
                self.stats['rebel_relations_extracted'] += num_added
                self.stats['instances_with_rebel'] += 1

        # Store total unique entities
        self.stats['rebel_entities_extracted'] = len(unique_entities)

        # Save enriched graph
        print(f"\nSaving enriched graph...")
        graph.serialize(destination=graph_path, format="turtle")

        final_triples = len(graph)
        print(f"Graph enriched:")
        print(f"  Initial triples: {initial_triples}")
        print(f"  Final triples: {final_triples}")
        print(f"  REBEL relations added: {self.stats['rebel_relations_extracted']}")
        print(f"  Unique entities: {self.stats['rebel_entities_extracted']}")
        print(f"  Instances with relations: {self.stats['instances_with_rebel']}")

    def run(self):
        """Main execution"""
        print("\n" + "=" * 50)
        print("Adding REBEL Relations to Graph")
        print("=" * 50)

        # Load extractions
        extractions = self.load_extractions()

        if not extractions:
            print("No extractions found. Exiting.")
            return

        # Process each graph
        for graph_path in self.graph_paths:
            self.add_to_graph(graph_path, extractions)

        # Print final statistics
        print("\n" + "=" * 50)
        print("Final Statistics")
        print("=" * 50)
        print(f"Relations added: {self.stats['rebel_relations_extracted']}")
        print(f"Unique entities: {self.stats['rebel_entities_extracted']}")
        print(f"Instances with relations: {self.stats['instances_with_rebel']}")


if __name__ == "__main__":
    # Input file
    EXTRACTIONS_FILE = OUTPUT_DIR / "rebel_extractions.json"

    if not Path(EXTRACTIONS_FILE).exists():
        print(f"Error: Extractions file not found: {EXTRACTIONS_FILE}")
        print("Please run extract_rebel_relations.py first")
        exit(1)

    # Initialize adder
    adder = REBELGraphAdder(extractions_file=EXTRACTIONS_FILE)

    # Add relations to graph
    adder.run()

    print("\n=== Complete ===")