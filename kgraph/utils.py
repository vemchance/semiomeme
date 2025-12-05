import pandas as pd
import re
from rdflib import Literal, RDF, RDFS, XSD


def clean_uri_string(text):
    """Clean text for use as a URI, normalising case to ensure consistency."""
    if not text:
        return "unknown"

    text = text.lower()
    text = text.replace(" ", "_")

    # Allow alphanumeric, underscore only
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', text)

    # Remove consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)

    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')

    # Ensure we have something
    if not cleaned:
        cleaned = "item"

    return cleaned


def resolve_relationship_dynamic(relationship_text, rebel_mappings=None):
    """Simplify relationship text to a standardised predicate."""
    relationship_text = relationship_text.lower().strip()

    # Use data-driven mappings if available
    if rebel_mappings and relationship_text in rebel_mappings:
        mapped = rebel_mappings[relationship_text]
        return mapped.replace(' ', '_')

    simplified = re.sub(r'[^a-z0-9]', '_', relationship_text)
    simplified = re.sub(r'_+', '_', simplified)
    return simplified.strip('_')


def clean_and_add_temporal_data(g, entity, row, smo_namespace):
    """Add temporal data using proper semantic types only."""

    if not pd.isna(row['Year']):
        year_val = str(row['Year']).strip()
        if year_val.lower() != 'unknown' and re.match(r'^(19|20)\d{2}$', year_val):
            g.add((entity, smo_namespace.yearCreated, Literal(year_val, datatype=XSD.gYear)))

    if not pd.isna(row['Origin']):
        origin_val = str(row['Origin']).strip()
        if origin_val.lower() != 'unknown':
            g.add((entity, smo_namespace.originDescription, Literal(origin_val)))


def infer_entity_type_from_context(entity_name, relation, context_text=""):
    """All REBEL-extracted entities are SMO.Entity."""
    from rdflib import Namespace
    SMO = Namespace("http://semiomeme.org/ontology/")
    return SMO.Entity

def apply_pattern_based_corrections(g, entity_registry):
    """Apply targeted corrections using patterns. Call this after process_meme_data()."""
    from rdflib import Namespace
    SMO = Namespace("http://semiomeme.org/ontology/")

    corrections = 0
    max_corrections = 100

    for entity_uri, entity_data in entity_registry.items():
        if corrections >= max_corrections:
            break

        entity_node = entity_data['uri']
        current_type = entity_data['type']

        labels = list(g.objects(entity_node, RDFS.label))
        if not labels:
            continue

        label_lower = str(labels[0]).lower()

        # Fix meme titles incorrectly typed as persons
        if (str(current_type).endswith('PersonEntry') and
                re.search(r'\b(thread|moment|style|text|effect|ing|ment)$', label_lower)):

            g.remove((entity_node, RDF.type, current_type))
            g.add((entity_node, RDF.type, SMO.MemeEntry))
            entity_data['type'] = SMO.MemeEntry
            corrections += 1

        # Fix events incorrectly typed as tags
        elif (str(current_type).endswith('Tag') and
              re.search(r'\b(attack|shooting|bombing|trial|election)\b', label_lower)):

            g.remove((entity_node, RDF.type, current_type))
            g.add((entity_node, RDF.type, SMO.EventEntry))
            entity_data['type'] = SMO.EventEntry
            corrections += 1

    if corrections > 0:
        print(f"Applied {corrections} pattern-based corrections")
    return corrections