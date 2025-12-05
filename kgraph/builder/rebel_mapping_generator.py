import os
import pickle
import json
from collections import Counter
from rdflib import Graph, Namespace
from difflib import SequenceMatcher
from config.namespaces import SMO, EX



class REBELMappingGenerator:
    """
    Analyses existing graphs to discover REBEL relation patterns and generate
    standardised mappings using purely data-driven approach.
    """

    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "rebel_mappings.pkl")
        self.analysis_file = os.path.join(cache_dir, "rebel_mapping_analysis.json")

        # Namespaces
        self.SMO = SMO
        self.EX = EX

        os.makedirs(cache_dir, exist_ok=True)

        # Results
        self.rebel_relations = Counter()
        self.canonical_relations = {}
        self.generated_mappings = {}

    def analyse_graph(self, ttl_file_path):
        """
        Analyse existing graphs to find all REBEL-extracted relations.
        """
        print(f"Loading graphs from {ttl_file_path}...")
        g = Graph()
        g.parse(ttl_file_path, format='turtle')
        print(f"Loaded {len(g)} triples")

        # Find all REBEL relations by checking extractedBy annotation
        print("Finding REBEL-extracted relations...")

        for prop, p, extractor in g.triples((None, self.SMO.extractedBy, None)):
            if str(extractor) == "REBEL":
                # Count usage of this property
                usage_count = len(list(g.triples((None, prop, None))))
                prop_name = str(prop).split('/')[-1]
                self.rebel_relations[prop_name] = usage_count

        print(f"Found {len(self.rebel_relations)} unique REBEL relations")
        print(f"Total REBEL relation instances: {sum(self.rebel_relations.values())}")

        # Show top relations
        print("\nTop 20 REBEL relations:")
        for rel, count in self.rebel_relations.most_common(20):
            print(f"  {rel}: {count}")

    def find_canonical_relations(self, min_frequency=5):
        """
        Identify canonical relations based purely on frequency.
        """
        print(f"\nFinding canonical relations (min frequency: {min_frequency})...")

        self.canonical_relations = {
            rel: count for rel, count in self.rebel_relations.items()
            if count >= min_frequency
        }

        print(f"Identified {len(self.canonical_relations)} canonical relations")

        # Show canonical relations
        print("\nCanonical relations:")
        for rel in sorted(self.canonical_relations.keys(),
                          key=lambda x: self.canonical_relations[x], reverse=True):
            print(f"  {rel}: {self.canonical_relations[rel]}")

    def generate_mappings_from_similarity(self, similarity_threshold=0.6):
        """
        Generate mappings by finding similar relations to canonical forms.
        """
        print(f"\nGenerating mappings (similarity threshold: {similarity_threshold})...")

        self.generated_mappings = {}

        for relation, count in self.rebel_relations.items():
            if relation not in self.canonical_relations:
                # Find most similar canonical relation
                best_match = self._find_most_similar(relation, self.canonical_relations.keys(),
                                                     similarity_threshold)
                if best_match:
                    # Convert to natural language for mapping
                    natural_form = relation.replace('_', ' ')
                    canonical_form = best_match.replace('_', ' ')
                    self.generated_mappings[natural_form] = canonical_form

        print(f"Generated {len(self.generated_mappings)} mappings")

        # Show sample mappings
        print("\nSample mappings:")
        sample_items = list(self.generated_mappings.items())[:15]
        for natural, canonical in sorted(sample_items):
            old_count = self.rebel_relations.get(natural.replace(' ', '_'), 0)
            new_count = self.canonical_relations.get(canonical.replace(' ', '_'), 0)
            print(f"  '{natural}' ({old_count}) -> '{canonical}' ({new_count})")

    def _find_most_similar(self, relation, canonical_relations, threshold=0.6):
        """
        Find most similar canonical relation using string similarity.
        """
        best_match = None
        best_score = 0

        for canonical in canonical_relations:
            score = SequenceMatcher(None, relation, canonical).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = canonical

        return best_match

    def save_mappings(self):
        """
        Save generated mappings to cache file for use by graphs builder.
        """
        mapping_data = {
            'mappings': self.generated_mappings,
            'canonical_relations': self.canonical_relations,
            'all_relation_stats': dict(self.rebel_relations),
            'generation_method': 'frequency_and_similarity',
            'total_relations_found': len(self.rebel_relations),
            'canonical_count': len(self.canonical_relations),
            'mappings_generated': len(self.generated_mappings)
        }

        # Save as pickle for fast loading
        with open(self.cache_file, 'wb') as f:
            pickle.dump(mapping_data, f)

        # Save human-readable analysis
        analysis_data = {
            'summary': {
                'total_rebel_relations': len(self.rebel_relations),
                'total_instances': sum(self.rebel_relations.values()),
                'canonical_relations': len(self.canonical_relations),
                'mappings_generated': len(self.generated_mappings)
            },
            'canonical_relations': dict(sorted(self.canonical_relations.items(),
                                               key=lambda x: x[1], reverse=True)),
            'generated_mappings': self.generated_mappings,
            'low_frequency_relations': {
                rel: count for rel, count in self.rebel_relations.items()
                if count < 5 and rel not in self.generated_mappings
            }
        }

        with open(self.analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        print(f"\nSaved mappings to {self.cache_file}")
        print(f"Saved analysis to {self.analysis_file}")

    def run_analysis(self, ttl_file_path, min_frequency=5, similarity_threshold=0.6):
        """
        Run complete analysis pipeline.
        """
        self.analyse_graph(ttl_file_path)
        self.find_canonical_relations(min_frequency)
        self.generate_mappings_from_similarity(similarity_threshold)
        self.save_mappings()

        print(f"\nAnalysis complete!")
        print(f"Found {len(self.rebel_relations)} unique REBEL relations")
        print(f"Identified {len(self.canonical_relations)} canonical forms")
        print(f"Generated {len(self.generated_mappings)} standardisation mappings")

        # Show consolidation impact
        original_relations = len(self.rebel_relations)
        final_relations = len(self.canonical_relations) + len([r for r in self.rebel_relations
                                                               if r not in self.canonical_relations
                                                               and r.replace('_', ' ') not in self.generated_mappings])
        print(f"Relation consolidation: {original_relations} -> {final_relations} "
              f"({((original_relations - final_relations) / original_relations * 100):.1f}% reduction)")


def load_rebel_mappings(cache_dir="cache"):
    """
    Load REBEL mappings from cache for use in graphs builder.
    Returns the mappings dict or None if not found.
    """
    cache_file = os.path.join(cache_dir, "rebel_mappings.pkl")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded {len(data['mappings'])} REBEL mappings from cache")
            print(f"Based on {data['total_relations_found']} relations, "
                  f"{data['canonical_count']} canonical forms")
            return data['mappings']
        except Exception as e:
            print(f"Error loading REBEL mappings: {e}")
            return None
    else:
        print("No REBEL mappings cache found - will use relations as extracted")
        return None