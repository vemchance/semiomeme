#!/usr/bin/env python3
"""
Detailed Graph Statistics
Focuses on entity/edge counts and extraction sources with proper REBEL categorization
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from rdflib import Graph, RDF, URIRef, Literal, Namespace
from pathlib import Path
from collections import defaultdict
import json

from config.namespaces import SMO, EX, SCHEMA, OWL

from config.config import META_CONFIG, CORPUS_CONFIG

meta_graph = META_CONFIG.DEFAULT_GRAPH # path to meta only graph
corpus_graph = CORPUS_CONFIG.INSTANCE_GRAPH_PATH # path to meta + corpus graph
instance_graph = CORPUS_CONFIG.INSTANCE_ONLY_GRAPH_PATH # instance only graph path


def analyze_graph_structure(g):
    """Analyze graph structure: nodes, edges, literals"""

    print("\n=== GRAPH STRUCTURE ===")

    # Separate URI objects from literal objects
    uri_objects = set()
    literal_objects = set()
    uri_triples = 0
    literal_triples = 0

    for s, p, o in g:
        if isinstance(o, URIRef):
            uri_objects.add(o)
            uri_triples += 1
        elif isinstance(o, Literal):
            literal_objects.add(o)
            literal_triples += 1

    # Calculate unique nodes (subjects + URI objects, deduplicated)
    all_subjects = set(g.subjects())
    all_nodes = all_subjects.union(uri_objects)

    stats = {
        'total_triples': len(g),
        'unique_subjects': len(all_subjects),
        'unique_uri_objects': len(uri_objects),
        'unique_literal_objects': len(literal_objects),
        'unique_nodes': len(all_nodes),  # Deduplicated subjects + URI objects
        'edges': uri_triples,  # Triples with URI objects
        'literal_triples': literal_triples,
        'uri_triple_percentage': (uri_triples / len(g)) * 100 if len(g) > 0 else 0
    }

    print(f"Total triples: {stats['total_triples']:,}")
    print(f"Unique subjects: {stats['unique_subjects']:,}")
    print(f"Unique URI objects: {stats['unique_uri_objects']:,}")
    print(f"Unique nodes (subjects + URI objects): {stats['unique_nodes']:,}")
    print(f"Edges (URI-to-URI triples): {stats['edges']:,}")
    print(f"Literal triples: {stats['literal_triples']:,}")
    print(f"URI triple percentage: {stats['uri_triple_percentage']:.1f}%")

    return stats


def analyze_entity_sources(g):
    """Analyze where entities come from, with detailed REBEL breakdown"""

    print("\n=== ENTITY EXTRACTION SOURCES ===")

    # Find all REBEL predicates from graph metadata
    rebel_predicates = set(g.subjects(SMO.extractedBy, Literal("REBEL")))

    # Separate REBEL predicates by source
    ocr_predicates = set()
    description_predicates = set()

    for pred in rebel_predicates:
        sources = list(g.objects(pred, SMO.extractedFromSource))
        if sources and "ocr_caption" in str(sources[0]):
            ocr_predicates.add(pred)
        else:
            # No extractedFromSource = from description
            description_predicates.add(pred)

    # Collect entities from OCR-extracted relations
    ocr_entities = set()
    ocr_relation_count = 0
    for pred in ocr_predicates:
        for s, o in g.subject_objects(pred):
            ocr_entities.add(s)
            if isinstance(o, URIRef):
                ocr_entities.add(o)
            ocr_relation_count += 1

    # Collect entities from description-extracted relations
    description_entities = set()
    description_relation_count = 0
    for pred in description_predicates:
        for s, o in g.subject_objects(pred):
            description_entities.add(s)
            if isinstance(o, URIRef):
                description_entities.add(o)
            description_relation_count += 1

    # Total REBEL entities (combined, deduplicated)
    total_rebel_entities = ocr_entities | description_entities
    total_rebel_relations = ocr_relation_count + description_relation_count

    # Other entity sources for context
    wikidata_entities = set()
    wikidata_edges = 0  # Count owl:sameAs triples
    for s, o in g.subject_objects(OWL.sameAs):
        if 'wikidata.org' in str(o):
            wikidata_entities.add(s)
            wikidata_edges += 1  # Each sameAs triple is an edge to Wikidata

    meme_instances = set(g.subjects(RDF.type, SMO.MemeInstance))

    kym_metadata_entities = set()
    for s, o in g.subject_objects(SMO.hasTag):
        kym_metadata_entities.add(o)
    for s, o in g.subject_objects(SMO.partOfSeries):
        kym_metadata_entities.add(o)
    for s, o in g.subject_objects(SMO.hasMemeType):
        kym_metadata_entities.add(o)
    for s, o in g.subject_objects(SMO.hasStatus):
        kym_metadata_entities.add(o)

    instances_with_ocr = set(g.subjects(SMO.hasOCRText, None))

    # Count all entity types in the graph
    all_entity_types = set()
    entity_type_counts = defaultdict(int)
    for s, o in g.subject_objects(RDF.type):
        if isinstance(o, URIRef):
            all_entity_types.add(o)
            entity_type_counts[o] += 1

    # Count specific KYM entries for reference
    kym_entries = set()
    entry_types = ['MemeEntry', 'PersonEntry', 'EventEntry',
                   'SubcultureEntry', 'WebsiteEntry']
    for entry_type in entry_types:
        kym_entries.update(g.subjects(RDF.type, SMO[entry_type]))

    stats = {
        'rebel_entities_total': len(total_rebel_entities),
        'rebel_entities_ocr': len(ocr_entities),
        'rebel_entities_description': len(description_entities),
        'rebel_relations_total': total_rebel_relations,
        'rebel_relations_ocr': ocr_relation_count,
        'rebel_relations_description': description_relation_count,
        'rebel_predicates_total': len(rebel_predicates),
        'rebel_predicates_ocr': len(ocr_predicates),
        'rebel_predicates_description': len(description_predicates),
        'wikidata_linked': len(wikidata_entities),
        'wikidata_nodes': len(wikidata_entities),  # Same as linked, for clarity
        'wikidata_edges': wikidata_edges,  # Number of sameAs links
        'meme_instances': len(meme_instances),
        'instances_with_ocr': len(instances_with_ocr),
        'kym_metadata_entities': len(kym_metadata_entities),
        'kym_entries': len(kym_entries),
        'all_entity_types': len(all_entity_types),
        'entity_type_counts': dict(entity_type_counts)
    }

    print(f"\n--- REBEL EXTRACTION BREAKDOWN ---")
    print(f"Entities extracted (total): {stats['rebel_entities_total']:,}")
    print(f"  From captions (OCR text): {stats['rebel_entities_ocr']:,}")
    print(f"  From descriptions: {stats['rebel_entities_description']:,}")
    print(f"\nRelations extracted (from REBEL, all): {stats['rebel_relations_total']:,}")
    print(f"  From captions (OCR): {stats['rebel_relations_ocr']:,}")
    print(f"  From descriptions: {stats['rebel_relations_description']:,}")
    print(f"\nUnique REBEL predicates: {stats['rebel_predicates_total']:,}")
    print(f"  OCR-sourced: {stats['rebel_predicates_ocr']:,}")
    print(f"  Description-sourced: {stats['rebel_predicates_description']:,}")

    print(f"\n--- OTHER ENTITY SOURCES ---")
    print(f"Wikidata nodes (entities with sameAs links): {stats['wikidata_nodes']:,}")
    print(f"Wikidata edges (owl:sameAs triples): {stats['wikidata_edges']:,}")
    print(f"Meme instances (from images): {stats['meme_instances']:,}")
    print(f"Instances with OCR text: {stats['instances_with_ocr']:,}")
    print(f"KYM metadata entities (tags/series/types): {stats['kym_metadata_entities']:,}")
    print(f"KYM entries (base): {stats['kym_entries']:,}")

    print(f"\n--- ENTITY TYPES ---")
    print(f"Total entity types in graph: {stats['all_entity_types']:,}")
    print(f"\nTop entity types by count:")
    for entity_type, count in sorted(stats['entity_type_counts'].items(),
                                     key=lambda x: x[1], reverse=True)[:15]:
        type_name = str(entity_type).split('/')[-1].split('#')[-1]
        print(f"  {type_name}: {count:,}")

    return stats


def analyze_relations(g):
    """Analyze relation types and sources (edges = URI-to-URI triples)"""

    print("\n=== RELATION ANALYSIS ===")

    # Count by predicate namespace
    smo_relations = 0
    rdf_relations = 0
    rdfs_relations = 0
    owl_relations = 0
    other_relations = 0

    predicate_counts = defaultdict(int)

    # Count REBEL vs non-REBEL relations
    rebel_predicates = set(g.subjects(SMO.extractedBy, Literal("REBEL")))
    rebel_relation_count = 0
    non_rebel_relation_count = 0

    # Also track URI-to-URI edges specifically
    total_edges = 0

    for s, p, o in g:
        pred_str = str(p)
        predicate_counts[p] += 1

        # Count edges (URI-to-URI triples only)
        if isinstance(o, URIRef):
            total_edges += 1

        if 'semiomeme.org/ontology' in pred_str:
            smo_relations += 1
        elif 'www.w3.org/1999/02/22-rdf' in pred_str:
            rdf_relations += 1
        elif 'www.w3.org/2000/01/rdf-schema' in pred_str:
            rdfs_relations += 1
        elif 'www.w3.org/2002/07/owl' in pred_str:
            owl_relations += 1
        else:
            other_relations += 1

        # Count REBEL vs non-REBEL (edges only)
        if isinstance(o, URIRef):
            if p in rebel_predicates:
                rebel_relation_count += 1
            else:
                non_rebel_relation_count += 1

    # Top SMO properties
    print("\nTop 10 SMO properties:")
    smo_props = [(p, c) for p, c in predicate_counts.items()
                 if 'semiomeme.org/ontology' in str(p)]
    for prop, count in sorted(smo_props, key=lambda x: x[1], reverse=True)[:10]:
        prop_name = str(prop).split('/')[-1]
        print(f"  smo:{prop_name}: {count:,}")

    stats = {
        'smo_relations': smo_relations,
        'rdf_relations': rdf_relations,
        'rdfs_relations': rdfs_relations,
        'owl_relations': owl_relations,
        'other_relations': other_relations,
        'unique_predicates': len(predicate_counts),
        'total_edges': total_edges,  # URI-to-URI triples (edges in graph theory)
        'rebel_edges': rebel_relation_count,  # REBEL-extracted edges
        'kym_edges': non_rebel_relation_count  # Structured KYM edges
    }

    print(f"\nRelation counts by namespace:")
    print(f"  SMO (domain): {smo_relations:,}")
    print(f"  RDF (type): {rdf_relations:,}")
    print(f"  RDFS (label/comment): {rdfs_relations:,}")
    print(f"  OWL (sameAs): {owl_relations:,}")
    print(f"  Other: {other_relations:,}")

    print(f"\nEdge counts (URI-to-URI relations):")
    print(f"  Total edges: {stats['total_edges']:,}")
    print(f"  REBEL-extracted edges: {stats['rebel_edges']:,}")
    print(f"  KYM structured edges: {stats['kym_edges']:,}")
    print(f"  Unique predicates: {stats['unique_predicates']:,}")

    return stats


def generate_paper_table(graph_path):
    """Generate complete statistics for paper table"""

    print(f"\nLoading {Path(graph_path).name}...")
    g = Graph()
    g.parse(graph_path, format="turtle")

    structure = analyze_graph_structure(g)
    entities = analyze_entity_sources(g)
    relations = analyze_relations(g)

    # Compile for paper
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER TABLE")
    print("=" * 60)

    print("\nGraph Structure:")
    print(f"  Nodes (unique entities): {structure['unique_nodes']:,}")
    print(f"  Edges (URI-to-URI relations): {relations['total_edges']:,}")
    print(f"    KYM edges: {relations['kym_edges']:,}")
    print(f"    REBEL edges: {relations['rebel_edges']:,}")
    print(f"  Total triples (incl. literals): {structure['total_triples']:,}")
    print(f"  Unique predicates: {relations['unique_predicates']:,}")

    print("\nEntity Types:")
    print(f"  Total entity types modeled: {entities['all_entity_types']:,}")

    print("\nEntity Extraction:")
    print(f"  Entities extracted (total): {entities['rebel_entities_total']:,}")
    print(f"    From captions (OCR text): {entities['rebel_entities_ocr']:,}")
    print(f"    From descriptions: {entities['rebel_entities_description']:,}")
    print(f"  Meme instances (from images): {entities['meme_instances']:,}")
    print(f"  Wikidata nodes (with sameAs): {entities['wikidata_nodes']:,}")
    print(f"  Wikidata edges (sameAs links): {entities['wikidata_edges']:,}")
    print(f"  KYM metadata entities: {entities['kym_metadata_entities']:,}")

    print("\nRelation Extraction:")
    print(f"  Relations extracted (from REBEL, all): {entities['rebel_relations_total']:,}")
    print(f"    From captions (OCR): {entities['rebel_relations_ocr']:,}")
    print(f"    From descriptions: {entities['rebel_relations_description']:,}")
    print(f"  Structured KYM relations: {relations['kym_edges']:,}")
    print(f"  Total edges: {relations['total_edges']:,}")

    # Save JSON
    all_stats = {
        'file': Path(graph_path).name,
        'structure': structure,
        'entities': entities,
        'relations': relations
    }

    output = f"detailed_stats_{Path(graph_path).stem}.json"
    with open(output, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nDetailed stats saved to: {output}")

    return all_stats


if __name__ == "__main__":
    # Specify your graph
    graph = str(instance_graph)

    generate_paper_table(graph)
