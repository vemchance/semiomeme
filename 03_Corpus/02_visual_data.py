# add_visual_elements.py
"""
Post-processing script to add visual elements from Google Cloud Vision API web detection
to existing RDF graphs. Adds web entities, best guess labels, and web detection metadata.
"""

import json
import pandas as pd
from pathlib import Path
from rdflib import Graph, Literal, URIRef, Namespace, XSD
from tqdm import tqdm
import os
import ast

# Import settings and namespaces
from config.settings import (
    OUTPUT_DIR,
    INSTANCE_GRAPH_PATH,
    INSTANCE_ONLY_GRAPH_PATH
)
from config.namespaces import SMO, EX, SCHEMA, RDFS


class VisualElementEnricher:
    """Adds visual elements from web detection to existing RDF instance graphs"""

    def __init__(self, web_detection_csv: str, graph_paths=None):
        """
        Initialize with web detection CSV and paths to graphs to enrich

        Args:
            web_detection_csv: Path to CSV file with web detection results
            graph_paths: List of paths to RDF graphs to enrich.
                        If None, uses both default graphs from settings
        """
        self.web_detection_csv = web_detection_csv

        if graph_paths is None:
            # Use both graphs by default
            self.graph_paths = [
                str(INSTANCE_GRAPH_PATH),  # Combined graph
                str(INSTANCE_ONLY_GRAPH_PATH)  # Instance-only graph
            ]
        else:
            self.graph_paths = graph_paths

        self.visual_data = {}
        self.stats = {
            'total_instances': 0,
            'instances_with_visual': 0,
            'instances_without_visual': 0,
            'web_entities_added': 0,
            'best_guesses_added': 0,
            'matching_images_added': 0,
            'errors_found': 0
        }

    def load_visual_data(self):
        """Load visual detection data from CSV file"""
        print(f"\n=== Loading Visual Detection Data ===")
        print(f"Loading: {self.web_detection_csv}")

        if not Path(self.web_detection_csv).exists():
            print(f"Error: CSV file not found: {self.web_detection_csv}")
            return

        try:
            df = pd.read_csv(self.web_detection_csv)
            print(f"Loaded {len(df)} entries")

            # Process each row
            for _, row in df.iterrows():
                filename = row['filename']

                # Skip if there's an error
                if pd.notna(row['error']) and row['error']:
                    self.stats['errors_found'] += 1
                    continue

                visual_info = {}

                # Parse web entities (stored as JSON string)
                if pd.notna(row['web_entities']):
                    try:
                        web_entities = ast.literal_eval(row['web_entities'])
                        visual_info['web_entities'] = web_entities
                    except:
                        print(f"  Warning: Could not parse web_entities for {filename}")

                # Parse best guess labels
                if pd.notna(row['best_guess_labels']):
                    try:
                        best_guesses = ast.literal_eval(row['best_guess_labels'])
                        visual_info['best_guess_labels'] = best_guesses
                    except:
                        print(f"  Warning: Could not parse best_guess_labels for {filename}")

                # Parse full matching images
                if pd.notna(row['full_matching_images']):
                    try:
                        full_matches = ast.literal_eval(row['full_matching_images'])
                        visual_info['full_matching_images'] = full_matches
                    except:
                        pass

                # Parse visually similar images
                if pd.notna(row['visually_similar_images']):
                    try:
                        similar_images = ast.literal_eval(row['visually_similar_images'])
                        visual_info['visually_similar_images'] = similar_images
                    except:
                        pass

                # Parse pages with matching images
                if pd.notna(row['pages_with_matching_images']):
                    try:
                        pages = ast.literal_eval(row['pages_with_matching_images'])
                        visual_info['pages_with_matching_images'] = pages
                    except:
                        pass

                # Add timestamp and counts
                visual_info['timestamp'] = row['timestamp']
                visual_info['num_web_entities'] = row['num_web_entities']

                # Store the data
                if visual_info:
                    self.visual_data[filename] = visual_info

            print(f"Successfully processed {len(self.visual_data)} entries")
            print(f"Errors found: {self.stats['errors_found']}")

            # Show sample
            if self.visual_data:
                sample_file = list(self.visual_data.keys())[0]
                sample_data = self.visual_data[sample_file]
                print(f"\nSample entry ({sample_file}):")
                if 'web_entities' in sample_data and sample_data['web_entities']:
                    top_entities = sample_data['web_entities'][:3]
                    for entity in top_entities:
                        print(f"  - {entity['description']}: {entity['score']:.2f}")
                if 'best_guess_labels' in sample_data:
                    for guess in sample_data['best_guess_labels'][:1]:
                        print(f"  Best guess: {guess['label']}")

        except Exception as e:
            print(f"Error loading visual data: {e}")

    def enrich_graph(self, graph_path: str):
        """Add visual elements to a single graph"""
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
        entities_added = 0
        guesses_added = 0
        matches_added = 0
        not_found_count = 0

        for instance, image_path in tqdm(instances, desc="Adding visual elements"):
            self.stats['total_instances'] += 1

            # Get filename from image path
            filename = str(image_path)

            # Check if we have visual data for this file
            if filename in self.visual_data:
                visual_info = self.visual_data[filename]

                # Add web entities
                if 'web_entities' in visual_info:
                    for entity in visual_info['web_entities']:
                        # Create a blank node for each entity
                        entity_node = URIRef(EX[f"web_entity_{hash(entity['entity_id']) & 0xFFFFFFFF}"])

                        # Add entity properties
                        graph.add((instance, SMO.hasWebEntity, entity_node))
                        graph.add((entity_node, RDF.type, SMO.WebEntity))

                        if entity['description']:
                            graph.add((entity_node, RDFS.label, Literal(entity['description'])))

                        graph.add((entity_node, SMO.entityId, Literal(entity['entity_id'])))
                        graph.add((entity_node, SMO.confidenceScore,
                                   Literal(entity['score'], datatype=XSD.float)))

                        entities_added += 1

                # Add best guess labels
                if 'best_guess_labels' in visual_info:
                    for guess in visual_info['best_guess_labels']:
                        graph.add((instance, SMO.hasBestGuessLabel, Literal(guess['label'])))
                        if guess.get('language_code'):
                            graph.add((instance, SMO.labelLanguage, Literal(guess['language_code'])))
                        guesses_added += 1

                # Add counts of matching images
                if 'full_matching_images' in visual_info and visual_info['full_matching_images']:
                    graph.add((instance, SMO.fullMatchingImageCount,
                               Literal(len(visual_info['full_matching_images']), datatype=XSD.integer)))
                    matches_added += 1

                if 'visually_similar_images' in visual_info and visual_info['visually_similar_images']:
                    graph.add((instance, SMO.visuallySimilarImageCount,
                               Literal(len(visual_info['visually_similar_images']), datatype=XSD.integer)))

                # Add web detection timestamp
                if 'timestamp' in visual_info:
                    graph.add((instance, SMO.webDetectionTimestamp,
                               Literal(visual_info['timestamp'], datatype=XSD.dateTime)))

                # Add number of web entities detected
                if 'num_web_entities' in visual_info:
                    graph.add((instance, SMO.webEntityCount,
                               Literal(visual_info['num_web_entities'], datatype=XSD.integer)))

                self.stats['instances_with_visual'] += 1
            else:
                not_found_count += 1
                self.stats['instances_without_visual'] += 1

        # Update statistics
        self.stats['web_entities_added'] += entities_added
        self.stats['best_guesses_added'] += guesses_added
        self.stats['matching_images_added'] += matches_added

        # Save enriched graph
        if entities_added > 0 or guesses_added > 0:
            print(f"\nSaving enriched graph...")
            graph.serialize(destination=graph_path, format="turtle")

            final_triples = len(graph)
            print(f"Graph enriched:")
            print(f"  Initial triples: {initial_triples}")
            print(f"  Final triples: {final_triples}")
            print(f"  Web entities added: {entities_added}")
            print(f"  Best guesses added: {guesses_added}")
            print(f"  Matching image counts added: {matches_added}")
            print(f"  No visual data found: {not_found_count}")
        else:
            print("No changes made to graph")

    def run(self):
        """Main execution pipeline"""
        print("\n" + "=" * 50)
        print("Visual Element Enrichment Pipeline")
        print("=" * 50)

        # Load visual data
        self.load_visual_data()

        if not self.visual_data:
            print("\nNo visual data loaded. Exiting.")
            return

        # Process each graph
        for graph_path in self.graph_paths:
            self.enrich_graph(graph_path)

        # Print final statistics
        self.print_statistics()

    def print_statistics(self):
        """Print enrichment statistics"""
        print("\n" + "=" * 50)
        print("Visual Enrichment Statistics")
        print("=" * 50)
        print(f"Total instances processed: {self.stats['total_instances']}")
        print(f"Instances with visual data: {self.stats['instances_with_visual']}")
        print(f"Instances without visual data: {self.stats['instances_without_visual']}")
        print(f"Web entities added: {self.stats['web_entities_added']}")
        print(f"Best guess labels added: {self.stats['best_guesses_added']}")
        print(f"Matching image counts added: {self.stats['matching_images_added']}")
        print(f"Errors in source data: {self.stats['errors_found']}")

        if self.stats['total_instances'] > 0:
            coverage = (self.stats['instances_with_visual'] / self.stats['total_instances']) * 100
            print(f"Visual data coverage: {coverage:.1f}%")


if __name__ == "__main__":
    # Configuration - modify this path to your web detection CSV file
    WEB_DETECTION_CSV = "path/to/web_detection.csv"  # UPDATE THIS PATH

    # Or use a path from settings if you want to add it there
    # from config.settings import WEB_DETECTION_CSV

    # Initialize enricher with CSV file and default graphs from settings
    enricher = VisualElementEnricher(
        web_detection_csv=WEB_DETECTION_CSV
        # graph_paths=None uses default graphs from settings
    )

    # Run enrichment
    enricher.run()