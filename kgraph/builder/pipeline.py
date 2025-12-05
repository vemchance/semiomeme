import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD
import ast
from tqdm import tqdm
import os
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

from kgraph.entities.wiki_enrichment import WikidataEntityLinker, LocalWikidataLinker
from kgraph.entities.ontology_mapper import OntologyMapper
from kgraph.builder.rebel_extractor import REBELTripleExtractor, process_rebel_relations
from kgraph.utils import (
    clean_uri_string, clean_and_add_temporal_data
)
from kgraph.entities.entity_resolver import EntityTypeResolver
from kgraph.builder.rebel_mapping_generator import load_rebel_mappings
from config.namespaces import SMO, EX, SCHEMA
from kgraph.entities.add_entity_to_graph import add_entity_to_graph


class MemeGraphPipeline:
    def __init__(self, config=None):
        self.config = config

    def process_meme_data(self, file_path, batch_size=None, test_mode=False, sample_size=100, cache_dir="cache",
                          enable_rebel=True, use_local_wikidata=True, enrich_rebel=False):
        """Process meme data with hierarchical ontology approach and REBEL extraction."""

        # Create cache directory if it doesn't exist
        if not os.path.isabs(cache_dir):
            project_root = Path(__file__).parent.parent.parent
            cache_dir = str(project_root / cache_dir)


        # Load REBEL mappings if available (only when REBEL is enabled)
        rebel_mappings = None
        if enable_rebel:
            rebel_mappings = load_rebel_mappings(cache_dir)
            if rebel_mappings:
                print(f"Using {len(rebel_mappings)} cached REBEL relation mappings")
            else:
                print("No REBEL mappings extracting relations as-is")
                print(
                    "Either run rebel_mapping_generator.py, or too few semantically similar relation mappings exist to use")


        # Choose entity linker based on configuration
        if use_local_wikidata:
            print("Using local Wikidata linker...")
            entity_linker = LocalWikidataLinker(
                data_dir=os.path.join(cache_dir, "wikidata_local"),
                namespace=str(EX)
            )
        else:
            print("Using online Wikidata linker...")
            entity_linker = WikidataEntityLinker(
                cache_file=os.path.join(cache_dir, "entity_links.pkl"),
                namespace=str(EX)
            )

        ontology_mapper = OntologyMapper(entity_linker)
        entity_resolver = EntityTypeResolver()

        # Initialise REBEL extractor if enabled
        rebel_extractor = None
        if enable_rebel:
            print("Initialising REBEL triple extractor...")
            rebel_extractor = REBELTripleExtractor()

        # Load data
        print(f"Reading data from {file_path}...")
        data = pd.read_csv(file_path)
        print(f"Loaded {len(data)} rows.")

        # Sample data if in test mode
        if test_mode:
            if sample_size >= len(data):
                print(f"Sample size {sample_size} exceeds data size {len(data)}. Using all data.")
            else:
                print(f"Taking a random sample of {sample_size} memes.")
                data = data.sample(n=sample_size, random_state=42)
                print(f"Sampled {len(data)} rows.")

        # Initialise RDF graphs
        g = Graph()

        # Bind namespaces
        ontology_mapper.bind_namespaces(g)
        g.bind("smo", SMO)
        g.bind("ex", EX)
        g.bind("schema", SCHEMA)

        # Use ontology mapper to define class hierarchies
        ontology_mapper.define_class_hierarchies(g)

        # Define property domains and ranges using ontology properties
        g.add((SMO.hasTag, RDFS.domain, SMO.KYMEntry))
        g.add((SMO.hasTag, RDFS.range, SMO.Tag))
        g.add((SMO.hasMemeType, RDFS.domain, SMO.MemeEntry))
        g.add((SMO.hasMemeType, RDFS.range, SMO.MemeType))
        g.add((SMO.hasStatus, RDFS.domain, SMO.KYMEntry))
        g.add((SMO.hasStatus, RDFS.range, SMO.Status))
        g.add((SMO.hasBadge, RDFS.domain, SMO.KYMEntry))
        g.add((SMO.hasBadge, RDFS.range, SMO.Badge))
        g.add((SMO.partOfSeries, RDFS.domain, SMO.KYMEntry))
        g.add((SMO.partOfSeries, RDFS.range, SMO.Series))
        g.add((SMO.yearCreated, RDFS.domain, SMO.KYMEntry))
        g.add((SMO.yearCreated, RDFS.range, XSD.gYear))
        g.add((SMO.originDescription, RDFS.domain, SMO.KYMEntry))
        g.add((SMO.originDescription, RDFS.range, XSD.string))
        g.add((SMO.mentions, RDFS.domain, SMO.KYMEntry))
        g.add((SMO.mentions, RDFS.range, SCHEMA.Thing))
        g.add((SMO.kymID, RDFS.domain, SMO.KYMEntry))
        g.add((SMO.kymID, RDFS.range, XSD.string))
        g.add((SMO.kymID, RDFS.label, Literal("KYM ID")))
        g.add((SMO.kymID, RDFS.comment, Literal("Unique identifier for KnowYourMeme.com classes and Corpus Bridge")))

        # Build the graphs
        print("Building knowledge graphs...")
        entity_registry = {}
        entity_types = set()
        total_rebel_relations = 0

        # Count total operations for accurate progress tracking
        enrichment_factor = 1 if enable_rebel and enrich_rebel else 0
        total_operations = len(data) * (3 + enrichment_factor if enable_rebel else 2)

        with tqdm(total=total_operations, desc="Building graphs") as pbar:
            for i, (_, row) in enumerate(data.iterrows()):
                # Skip rows with missing title
                if pd.isna(row['Title']):
                    pbar.update(3 + enrichment_factor if enable_rebel else 2)
                    continue

                entity_uri = clean_uri_string(row['Title'])
                entity = URIRef(EX[entity_uri])

                # Determine entity type from Entry Type field using ontology classes
                if 'Entry Type' in row and not pd.isna(row['Entry Type']):
                    entry_type = row['Entry Type'].strip().lower()

                    entry_class_map = {
                        'meme': SMO.MemeEntry,
                        'person': SMO.PersonEntry,
                        'event': SMO.EventEntry,
                        'culture': SMO.SubcultureEntry,  # Both map to same
                        'subculture': SMO.SubcultureEntry,  # Both map to same
                        'site': SMO.SiteEntry
                    }

                    entity_type_class = entry_class_map.get(entry_type, SMO.MemeEntry)
                else:
                    entity_type_class = SMO.MemeEntry

                # Add entity with ontology class
                g.add((entity, RDF.type, entity_type_class))
                entity_registry[entity_uri] = {'uri': entity, 'type': entity_type_class}
                entity_types.add(entity_type_class)

                if 'ID' in row and not pd.isna(row['ID']):
                    g.add((entity, SMO.kymID, Literal(str(row['ID']))))

                # Add title as label
                g.add((entity, RDFS.label, Literal(row['Title'])))

                if 'Views' in row and not pd.isna(row['Views']):
                    g.add((entity, SMO.viewCount, Literal(int(row['Views']), datatype=XSD.integer)))

                if 'Video Count' in row and not pd.isna(row['Video Count']):
                    g.add((entity, SMO.videoCount, Literal(int(row['Video Count']), datatype=XSD.integer)))

                if 'Photo Count' in row and not pd.isna(row['Photo Count']):
                    g.add((entity, SMO.photoCount, Literal(int(row['Photo Count']), datatype=XSD.integer)))

                if 'Comment Count' in row and not pd.isna(row['Comment Count']):
                    g.add((entity, SMO.commentCount, Literal(int(row['Comment Count']), datatype=XSD.integer)))

                # Add URLs and media references
                if 'KYM URL' in row and not pd.isna(row['KYM URL']):
                    g.add((entity, SMO.kymURL, Literal(str(row['KYM URL']))))

                if 'Main Image URL' in row and not pd.isna(row['Main Image URL']):
                    g.add((entity, SMO.mainImageURL, Literal(str(row['Main Image URL']))))

                # Add metadata
                if 'Region' in row and not pd.isna(row['Region']):
                    g.add((entity, SMO.region, Literal(str(row['Region']))))

                if 'Meta Description' in row and not pd.isna(row['Meta Description']):
                    g.add((entity, SMO.metaDescription, Literal(str(row['Meta Description']))))

                # Handle temporal data properly
                clean_and_add_temporal_data(g, entity, row, SMO)

                # Process Status using ontology
                if not pd.isna(row['Status']):
                    status_val = clean_uri_string(row['Status'].strip().lower())
                    status_node = add_entity_to_graph(
                        g, EX, status_val, SMO.Status, entity_registry, entity_types,
                        entity_resolver, ontology_mapper, label=row['Status'].strip()
                    )
                    g.add((entity, SMO.hasStatus, status_node))

                # Process Badges using ontology
                if 'Badges:' in row and not pd.isna(row['Badges:']):
                    badges_val = row['Badges:'].strip()
                    badges = [b.strip() for b in badges_val.split(',')] if ',' in badges_val else [badges_val]

                    for badge in badges:
                        if badge:
                            badge_uri = clean_uri_string(badge.lower())
                            badge_node = add_entity_to_graph(
                                g, EX, badge_uri, SMO.Badge, entity_registry, entity_types,
                                entity_resolver, ontology_mapper, label=badge
                            )
                            g.add((entity, SMO.hasBadge, badge_node))

                    # Change from using 'Type:' with comma split
                    # To using 'Meme Format' with pipe split:
                if 'Meme Format' in row and not pd.isna(row['Meme Format']):
                    formats_val = str(row['Meme Format']).strip()
                    meme_types = [t.strip() for t in formats_val.split('|')]

                    for meme_type in meme_types:
                        if meme_type:  # Skip empty strings
                            type_uri = clean_uri_string(meme_type.lower())
                            type_node = add_entity_to_graph(
                                g, EX, type_uri, SMO.MemeType, entity_registry, entity_types,
                                entity_resolver, ontology_mapper, label=meme_type
                            )
                            g.add((entity, SMO.hasMemeType, type_node))

                # Process tags using ontology
                if not pd.isna(row['Tags']):
                    tags_val = str(row['Tags']).strip()
                    if '|' in tags_val:
                        # New format: pipe-separated
                        tags = [tag.strip() for tag in tags_val.split('|')]
                    else:
                        # Try old format for compatibility
                        try:
                            tags = ast.literal_eval(tags_val)
                        except:
                            # Single tag
                            tags = [tags_val]

                    for tag in tags:
                        if tag:  # Skip empty strings
                            tag_uri = clean_uri_string(tag)
                            tag_node = add_entity_to_graph(
                                g, EX, tag_uri, SMO.Tag, entity_registry, entity_types,
                                entity_resolver, ontology_mapper, label=tag
                            )
                            g.add((entity, SMO.hasTag, tag_node))

                # Process series using ontology
                if not pd.isna(row['Part of a series on']):
                    series_uri = clean_uri_string(row['Part of a series on'])
                    series_node = add_entity_to_graph(
                        g, EX, series_uri, SMO.Series, entity_registry, entity_types,
                        entity_resolver, ontology_mapper, label=row['Part of a series on']
                    )
                    g.add((entity, SMO.partOfSeries, series_node))

                # Update progress for main entity processing
                pbar.update(1)

                # Now do enrichment for the main entity
                pbar.set_description(f"Enriching entity {i + 1}/{len(data)}")
                ontology_mapper.map_entity_to_ontologies(
                    g, entity_uri, entity_type_class, EX, row['Title']
                )

                # Update progress for enrichment
                pbar.update(1)

                # REBEL extraction phase
                if enable_rebel and rebel_extractor:
                    pbar.set_description(f"REBEL extraction {i + 1}/{len(data)}")

                    # Collect all text fields for this entity
                    text_fields = []
                    text_columns = ['Full Text']

                    for col in text_columns:
                        if col in row and not pd.isna(row[col]):
                            text_content = str(row[col]).strip()
                            if text_content and len(text_content) > 25:
                                text_fields.append(text_content)

                    # Extract relations from text if we have any substantial content
                    if text_fields:
                        try:
                            # Combine text fields for processing
                            combined_text = ' '.join(text_fields)

                            # Use chunked processing for longer texts
                            if len(combined_text.split()) > 400:
                                relations_list = rebel_extractor.process_full_text([combined_text])
                                if relations_list:
                                    rebel_relations = relations_list[0]
                                else:
                                    rebel_relations = []
                            else:
                                # Process directly for shorter texts
                                relations_batch = rebel_extractor.extract_triples_batch([combined_text],
                                                                                        show_progress=False)
                                if relations_batch:
                                    rebel_relations = relations_batch[0]
                                else:
                                    rebel_relations = []

                            # Process extracted relations using new ontology and mappings
                            if rebel_relations:
                                relations_added = process_rebel_relations(
                                    g, rebel_relations, entity, SMO, EX, entity_registry,
                                    entity_types, entity_resolver, ontology_mapper, rebel_mappings
                                )
                                total_rebel_relations += relations_added

                                # Enrich newly created entities from REBEL extraction if enabled
                                if enrich_rebel:
                                    newly_added_entities = []
                                    for subj, rel, obj in rebel_relations:
                                        subj_uri = clean_uri_string(subj)
                                        obj_uri = clean_uri_string(obj)

                                        if subj_uri in entity_registry:
                                            newly_added_entities.append((subj_uri, subj))
                                        if obj_uri in entity_registry:
                                            newly_added_entities.append((obj_uri, obj))

                                    # Remove duplicates
                                    newly_added_entities = list(set(newly_added_entities))

                                    # Enrich each newly added entity
                                    for entity_uri_new, original_label in newly_added_entities:
                                        if entity_uri_new in entity_registry:
                                            entity_type_new = entity_registry[entity_uri_new]['type']
                                            ontology_mapper.map_entity_to_ontologies(
                                                g, entity_uri_new, entity_type_new, EX, original_label
                                            )

                        except Exception as rebel_error:
                            print(f"REBEL extraction error for entity {entity_uri}: {rebel_error}")

                    # Update progress for REBEL processing
                    pbar.update(1)

                    # Update progress for REBEL enrichment (if enabled)
                    if enrich_rebel:
                        pbar.update(1)
                else:
                    # If REBEL is disabled, still update progress
                    if enable_rebel:
                        pbar.update(1)
                        if enrich_rebel:
                            pbar.update(1)

                pbar.set_description("Building graphs")

        # Print statistics
        print(f"Finished building graphs with {len(g)} triples.")
        print(f"Entity types created: {', '.join(str(t) for t in sorted(entity_types))}")
        print(f"Total distinct entities: {len(entity_registry)}")

        if enable_rebel:
            print(f"REBEL relations added: {total_rebel_relations}")
            if rebel_mappings:
                print(f"Used {len(rebel_mappings)} cached relation mappings for consistency")

        # Calculate and print enrichment summary
        enriched_count = 0
        wikidata_linked_count = 0

        for entity_uri, entity_data in entity_registry.items():
            entity_node = entity_data['uri']

            # Count entities with multiple types (enriched)
            entity_types_list = list(g.objects(entity_node, RDF.type))
            if len(entity_types_list) > 1:
                enriched_count += 1

            # Count entities with Wikidata links
            if list(g.objects(entity_node, OWL.sameAs)):
                wikidata_linked_count += 1

        print(f"Enrichment summary:")
        print(f"  - {enriched_count} entities enriched with additional types")
        print(f"  - {wikidata_linked_count} entities linked to Wikidata")

        if enriched_count > 0:
            enrichment_rate = (enriched_count / len(entity_registry)) * 100
            print(f"  - {enrichment_rate:.1f}% of entities were enriched")

        # Save entity linking cache
        entity_linker.save_cache()

        return g, entity_registry
