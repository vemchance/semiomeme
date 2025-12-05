import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import gc
import re
from rdflib import Literal, URIRef, XSD

# From your other modules
from kgraph.utils import (
    resolve_relationship_dynamic,
    infer_entity_type_from_context,
    clean_uri_string
)
from kgraph.entities.add_entity_to_graph import add_entity_to_graph


class REBELTripleExtractor:
    def __init__(self, model_name="Babelscape/rebel-large"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        # Default parameters
        self.default_batch_size = 16  # Reduced for memory efficiency
        self.max_seq_length = 768
        self.chunk_overlap = 128
        self.beam_width = 3

    def extract_relations_from_model_output(self, text):
        """Extract relations from model output text with improved parsing."""
        relations = []

        # Clean up the text first
        text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")

        # Extract using the correct token pattern
        relation, subject, obj = '', '', ''
        current = 'x'  # Current parsing state

        for token in text_replaced.split():
            if token == "<triplet>":
                current = 't'
                # Save previous triplet if complete
                if relation and subject and obj:
                    relations.append((subject.strip(), relation.strip(), obj.strip()))
                # Reset for new triplet
                subject, relation, obj = '', '', ''
            elif token == "<subj>":
                current = 's'
                # If we encounter <subj> directly after a relation was being built,
                # complete the previous triplet
                if relation and subject:
                    relations.append((subject.strip(), relation.strip(), obj.strip()))
                    # Reset only obj to start new relation
                    obj = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''  # Reset relation when starting a new object
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    obj += ' ' + token
                elif current == 'o':
                    relation += ' ' + token

        # Add the last triplet if complete
        if subject and relation and obj:
            relations.append((subject.strip(), relation.strip(), obj.strip()))

        return relations

    def extract_triples_batch(self, texts, batch_size=None, show_progress=True):
        """Extract triples from a batch of texts with better error handling."""
        if batch_size is None:
            batch_size = self.default_batch_size

        all_relations = []

        # Process texts in batches with improved memory management
        if show_progress:
            pbar = tqdm(total=len(texts), desc="Extracting Relationships")
        else:
            pbar = None
        try:
            i = 0
            current_batch_size = batch_size

            while i < len(texts):
                try:
                    # Safe memory cleanup without forced synchronisation
                    if torch.cuda.is_available():
                        try:
                            # Use a try/except block just for the memory cleanup
                            torch.cuda.empty_cache()
                            gc.collect()
                        except Exception as mem_e:
                            print(f"Warning: Memory cleanup issue (non-critical): {mem_e}")
                            # Continue execution - this error doesn't affect results

                    # Calculate how many texts to process in this iteration
                    end_idx = min(i + current_batch_size, len(texts))
                    batch_texts = texts[i:end_idx]

                    # Skip empty texts
                    batch_texts = [text for text in batch_texts if text and text.strip()]
                    if not batch_texts:
                        if pbar:
                            pbar.update(end_idx - i)
                        i = end_idx
                        continue

                    # Tokenise and generate with explicit error checking
                    model_inputs = self.tokenizer(
                        batch_texts,
                        max_length=self.max_seq_length,
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    ).to(self.device)

                    with torch.no_grad():
                        generated_tokens = self.model.generate(
                            **model_inputs,
                            max_length=self.max_seq_length,
                            length_penalty=0,
                            num_beams=self.beam_width,
                            num_return_sequences=1
                        )

                    # Process results immediately
                    decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

                    # Free memory explicitly before processing results
                    del model_inputs, generated_tokens
                    torch.cuda.empty_cache()

                    # Extract relations
                    batch_relations = []
                    for text, sentence_pred in zip(batch_texts, decoded_preds):
                        text_relations = self.extract_relations_from_model_output(sentence_pred)
                        batch_relations.append(text_relations)

                    all_relations.extend(batch_relations)

                    # Update progress and move to next batch
                    if pbar:
                        pbar.update(len(batch_texts))
                    i += len(batch_texts)

                except RuntimeError as e:
                    # Handle OOM errors by reducing batch size
                    if "out of memory" in str(e).lower():
                        # Reduce batch size for future iterations
                        original_size = current_batch_size
                        current_batch_size = max(1, current_batch_size // 2)

                        print(
                            f"\nCUDA OOM at batch {i // original_size}. Reducing batch size from {original_size} to {current_batch_size}")

                        # If batch size is already 1, add empty results and continue
                        if current_batch_size == 1 and original_size == 1:
                            print("Cannot reduce batch size further, adding empty results for this text")
                            all_relations.append([])
                            if pbar:
                                pbar.update(1)
                            i += 1
                    else:
                        # For non-OOM errors, raise to outer exception handler
                        raise

                    # Give the GPU a moment to recover
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()  # Force synchronisation
                            torch.cuda.empty_cache()
                            gc.collect()
                        except Exception as sync_e:
                            print(f"Warning: Error during GPU recovery: {sync_e}")

                except Exception as general_e:
                    # Handle any other exceptions
                    print(f"Error processing batch starting at index {i}: {general_e}")

                    # Add empty relations for this batch and continue
                    empty_count = min(current_batch_size, len(texts) - i)
                    all_relations.extend([[] for _ in range(empty_count)])
                    if pbar:
                        pbar.update(empty_count)
                    i += empty_count

        finally:
            if pbar:
                pbar.close()

        return all_relations

    def process_full_text(self, texts, chunk_size=512, chunk_overlap=100, show_progress=False):
        """Process full texts with chunking for longer texts."""
        all_chunks = []
        chunk_to_text_map = []

        # Create chunks for each text
        for i, text in enumerate(texts):
            if not text or not text.strip():
                chunk_to_text_map.append(i)
                all_chunks.append("")
                continue

            # Split into chunks if needed
            words = text.split()
            if len(words) <= chunk_size:
                all_chunks.append(text)
                chunk_to_text_map.append(i)
            else:
                # Create overlapping chunks
                start = 0
                while start < len(words):
                    end = min(start + chunk_size, len(words))
                    chunk = ' '.join(words[start:end])
                    all_chunks.append(chunk)
                    chunk_to_text_map.append(i)
                    start += (chunk_size - chunk_overlap)

        # Extract relations from all chunks
        chunk_relations = self.extract_triples_batch(all_chunks, show_progress=show_progress)

        # Combine relations by original text index
        results = [[] for _ in range(len(texts))]
        for chunk_idx, rels in enumerate(chunk_relations):
            if rels:  # Only if we got relations
                orig_idx = chunk_to_text_map[chunk_idx]
                results[orig_idx].extend(rels)

        return results

def process_rebel_relations(g, rebel_relations, main_entity_node, smo_namespace, ex_namespace, entity_registry,
                            entity_types, entity_resolver, ontology_mapper, rebel_mappings=None):
    """
    Process REBEL-extracted relations using new ontology structure with proper provenance and dual approach.
    """
    relations_added = 0

    # Temporal patterns to identify dates/years in extracted text
    temporal_patterns = {
        'year': r'^(19|20)\d{2}$',
        'full_date': r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}',
        'date_variant': r'\d{1,2}(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}'
    }

    for subject, relation, obj in rebel_relations:
        if not subject or not relation or not obj:
            continue

        # Clean up the extracted text
        subject = subject.strip()
        relation = relation.strip()
        obj = obj.strip()

        # Skip very short extractions
        if len(subject) < 2 or len(obj) < 2 or len(relation) < 2:
            continue

        # Check if object is temporal and should be converted to literal
        is_temporal_obj = False
        temporal_literal = None

        # Check for temporal patterns in object
        if re.match(temporal_patterns['year'], obj):
            is_temporal_obj = True
            temporal_literal = Literal(obj, datatype=XSD.gYear)
        elif any(re.search(pattern, obj.lower()) for pattern in
                 [temporal_patterns['full_date'], temporal_patterns['date_variant']]):
            is_temporal_obj = True
            temporal_literal = Literal(obj, datatype=XSD.string)

        # Handle temporal relations differently
        if is_temporal_obj:
            temporal_words = ['date', 'time', 'year', 'born', 'created', 'published', 'inception', 'period', 'start',
                              'end']
            is_temporal_relation = any(temporal_word in relation.lower() for temporal_word in temporal_words)

            if is_temporal_relation:
                # Map to appropriate temporal property
                if 'birth' in relation.lower() or 'born' in relation.lower():
                    temporal_prop = URIRef(smo_namespace['dateOfBirth'])
                    g.add((main_entity_node, temporal_prop, temporal_literal))
                    # Add provenance for temporal property
                    g.add((temporal_prop, smo_namespace.extractedBy, Literal("REBEL")))
                elif 'created' in relation.lower() or 'publication' in relation.lower() or 'inception' in relation.lower():
                    if temporal_literal.datatype == XSD.gYear:
                        temporal_prop = URIRef(smo_namespace['yearCreated'])
                        g.add((main_entity_node, temporal_prop, temporal_literal))
                    else:
                        temporal_prop = URIRef(smo_namespace['dateCreated'])
                        g.add((main_entity_node, temporal_prop, temporal_literal))
                    # Add provenance for temporal property
                    g.add((temporal_prop, smo_namespace.extractedBy, Literal("REBEL")))
                elif 'period' in relation.lower() or 'start' in relation.lower():
                    temporal_prop = URIRef(smo_namespace['temporalDescription'])
                    g.add((main_entity_node, temporal_prop, Literal(f"{relation}: {obj}")))
                    # Add provenance for temporal property
                    g.add((temporal_prop, smo_namespace.extractedBy, Literal("REBEL")))
                else:
                    # Generic temporal description
                    temporal_prop = URIRef(smo_namespace['temporalDescription'])
                    g.add((main_entity_node, temporal_prop, temporal_literal))
                    # Add provenance for temporal property
                    g.add((temporal_prop, smo_namespace.extractedBy, Literal("REBEL")))

                relations_added += 1
                continue  # Skip entity creation for temporal data

        # Handle as normal entity relation with dual approach
        predicate_name = resolve_relationship_dynamic(relation, rebel_mappings)
        predicate_uri = URIRef(smo_namespace[predicate_name])

        # CRITICAL: Add provenance tracking for REBEL-extracted predicate
        g.add((predicate_uri, smo_namespace.extractedBy, Literal("REBEL")))

        # Create subject entity with proper typing
        subject_uri = clean_uri_string(subject)
        subject_type = infer_entity_type_from_context(subject, relation)
        subject_node = add_entity_to_graph(
            g, ex_namespace, subject_uri, subject_type, entity_registry, entity_types,
            entity_resolver, ontology_mapper, label=subject
        )

        # Create object entity with proper typing
        obj_uri = clean_uri_string(obj)
        obj_type = infer_entity_type_from_context(obj, relation)
        obj_node = add_entity_to_graph(
            g, ex_namespace, obj_uri, obj_type, entity_registry, entity_types,
            entity_resolver, ontology_mapper, label=obj
        )

        # DUAL RELATIONSHIP APPROACH:

        g.add((subject_node, predicate_uri, obj_node))

        mentions_prop = URIRef(smo_namespace['mentions'])
        g.add((main_entity_node, mentions_prop, subject_node))
        g.add((main_entity_node, mentions_prop, obj_node))

        # Add provenance for mentions property (since these are also REBEL-derived)
        g.add((mentions_prop, smo_namespace.extractedBy, Literal("REBEL")))

        relations_added += 1

    return relations_added