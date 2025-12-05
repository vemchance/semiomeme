import os
import pickle
import re
import requests
import time
from typing import Dict, Optional, Any, Set
from rdflib import Graph, Literal, Namespace, URIRef, RDF, RDFS, XSD
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS, DCTERMS

import json
import gzip
from tqdm import tqdm
from urllib.parse import unquote


class WikidataEntityLinker:
    """
    Entity linker that uses ONLY Wikidata for external enrichment.
    No type inference - only owl:sameAs links.
    """

    def __init__(self, cache_file=None, namespace=None):
        """
        Initialize the WikidataEntityLinker.

        Args:
            cache_file: Path to a pickle file to store entity links
            namespace: The namespace to use for entity URIs
        """
        self.cache = {}
        self.cache_file = cache_file
        self._cache_modified = False
        self.namespace = namespace

        # Wikidata SPARQL endpoint
        self.wikidata_endpoint = "https://query.wikidata.org/sparql"

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests

        # Load cache if available
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    if isinstance(cache_data, dict):
                        if 'entity_cache' in cache_data:
                            self.cache = cache_data['entity_cache']
                        else:
                            self.cache = cache_data
                    print(f"Loaded {len(self.cache)} cached entity links")
            except Exception as e:
                print(f"Error loading cache: {e}")

    def _get_entity_uri(self, entity_name):
        """Generate an entity URI using the configured namespace."""
        clean_name = entity_name.replace(' ', '_')
        if self.namespace:
            return f"{self.namespace}{clean_name}"
        else:
            return f"http://example.org/entity/{clean_name}"

    def link_entity(self, entity_name, entity_type=None):
        """
        Link an entity using ONLY Wikidata - no type inference.

        Args:
            entity_name: The name of the entity to link
            entity_type: The primary type from KYM data (unused but kept for compatibility)

        Returns:
            Dictionary with entity linking information or None
        """
        if not entity_name:
            return None

        # Normalise entity name for caching
        cache_key = f"{entity_name}_{entity_type or 'none'}"

        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Try Wikidata lookup
        result = self._query_wikidata(entity_name)

        # If no Wikidata result, return None (no enrichment)
        if not result:
            result = None

        # Cache the result (including None results to avoid repeated queries)
        self.cache[cache_key] = result
        self._cache_modified = True

        return result

    def _query_wikidata(self, entity_name):
        """
        Query Wikidata SPARQL endpoint for entity information using optimised search strategies.
        Returns only the Wikidata URI for owl:sameAs linking.

        Args:
            entity_name: Name of the entity to search for

        Returns:
            Dictionary with Wikidata URI or None
        """
        # Try optimised search strategies in order of preference
        search_strategies = [
            self._exact_label_search,
            self._alias_search,
            self._variation_search  # Replaces the expensive contains/fuzzy searches
        ]

        for strategy in search_strategies:
            result = strategy(entity_name)
            if result:
                return result

        return None

    def _rate_limit(self):
        """Ensure rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def _execute_sparql_query(self, query, timeout=8):
        """Execute a SPARQL query with error handling and shorter timeout."""
        try:
            self._rate_limit()

            headers = {
                'User-Agent': 'SemiomemeKGBuilder/1.0',
                'Accept': 'application/sparql-results+json'
            }

            response = requests.get(
                self.wikidata_endpoint,
                params={'query': query, 'format': 'json'},
                headers=headers,
                timeout=timeout
            )

            if response.status_code != 200:
                return None

            data = response.json()
            results = data.get('results', {}).get('bindings', [])

            if results:
                best_result = results[0]
                wikidata_uri = best_result['item']['value']
                wikidata_id = wikidata_uri.split('/')[-1]

                return {
                    'wikidata_uri': wikidata_uri,
                    'wikidata_id': wikidata_id,
                    'source': 'wikidata'
                }

        except Exception as e:
            print(f"SPARQL query error for '{entity_name}': {e}")

        return None

    def _exact_label_search(self, entity_name):
        """Search for exact label match."""
        query = f"""
        SELECT ?item ?itemLabel WHERE {{
          ?item rdfs:label "{entity_name}"@en .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT 1
        """
        return self._execute_sparql_query(query)

    def _alias_search(self, entity_name):
        """Search using aliases (also known as) - optimised version."""
        query = f"""
        SELECT ?item ?itemLabel WHERE {{
          {{ ?item rdfs:label "{entity_name}"@en . }}
          UNION
          {{ ?item skos:altLabel "{entity_name}"@en . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT 1
        """
        return self._execute_sparql_query(query)

    def _variation_search(self, entity_name):
        """Search for specific variations using exact matches only."""
        # Generate a small set of high-confidence variations
        variations = self._generate_key_variations(entity_name)

        for variation in variations:
            # Use exact label search for each variation
            query = f"""
            SELECT ?item ?itemLabel WHERE {{
              {{ ?item rdfs:label "{variation}"@en . }}
              UNION
              {{ ?item skos:altLabel "{variation}"@en . }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
            }}
            LIMIT 1
            """
            result = self._execute_sparql_query(query, timeout=5)  # Shorter timeout for variations
            if result:
                return result

        return None

    def _generate_key_variations(self, entity_name):
        """Generate only the most likely variations to avoid expensive queries."""
        variations = []
        clean_name = entity_name.replace("_", " ").strip()

        # Don't add the original if it's the same as clean_name
        if clean_name != entity_name:
            variations.append(clean_name)

        # Handle initials patterns only for likely cases
        words = clean_name.split()
        if len(words) >= 2:
            first_word = words[0]

            # Pattern: "OJ Simpson" -> "O. J. Simpson" and "O.J. Simpson"
            if len(first_word) >= 2 and first_word.isupper() and '.' not in first_word:
                rest = " ".join(words[1:])
                # Add dotted version: "O.J. Simpson"
                dotted = ".".join(first_word) + "."
                variations.append(f"{dotted} {rest}")
                # Add spaced version: "O. J. Simpson"
                spaced = ". ".join(first_word) + "."
                variations.append(f"{spaced} {rest}")

            # Pattern: "O.J. Simpson" -> "OJ Simpson"
            elif '.' in first_word and len(first_word.replace('.', '')) >= 2:
                no_dots = first_word.replace('.', '')
                rest = " ".join(words[1:])
                variations.append(f"{no_dots} {rest}")

        # Only add common, high-confidence nickname mappings
        high_confidence_nicknames = {
            'Michael': 'Mike',
            'William': 'Bill',
            'Robert': 'Bob',
            'James': 'Jim',
            'Christopher': 'Chris',
            'Matthew': 'Matt'
        }

        if len(words) >= 2:
            first_name = words[0]
            if first_name in high_confidence_nicknames:
                nickname_version = clean_name.replace(first_name, high_confidence_nicknames[first_name], 1)
                variations.append(nickname_version)

        # Remove duplicates and original name
        unique_variations = []
        for var in variations:
            if var != entity_name and var not in unique_variations:
                unique_variations.append(var)

        # Limit to max 3 variations to avoid too many queries
        return unique_variations[:3]

    def save_cache(self):
        """Save the entity cache to disk."""
        if not self.cache_file or not self._cache_modified:
            return

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump({'entity_cache': self.cache}, f)
            print(f"Saved {len(self.cache)} entity links to cache")
            self._cache_modified = False
        except Exception as e:
            print(f"Error saving cache: {e}")


# Simple fallback for backward compatibility
class SpacyEntityLinker:
    """Deprecated - kept for backward compatibility but does nothing."""

    def __init__(self, cache_file=None):
        self.cache = {}
        self.cache_file = cache_file
        print("Warning: SpacyEntityLinker is deprecated. Use WikidataEntityLinker instead.")

    def link_entity(self, entity_name, entity_type=None):
        """Returns None to prevent any spaCy-based linking."""
        return None

    def save_cache(self):
        """No-op for compatibility."""
        pass

class LocalWikidataLinker:
    """
    Local Wikidata entity linker using downloaded labels dump.

    Downloads Wikidata labels dump and builds searchable index for fast
    entity linking without API rate limits.
    """

    def __init__(self, data_dir="cache/wikidata_local", namespace=None):
        """
        Initialize the local Wikidata linker.

        Args:
            data_dir: Directory to store Wikidata files
            namespace: Namespace for the knowledge graphs (for compatibility)
        """
        self.data_dir = data_dir
        self.namespace = namespace
        self.labels_file = os.path.join(data_dir, "latest-labels.json.gz")
        self.index_file = os.path.join(data_dir, "labels_index.pkl")

        # Create data directory
        os.makedirs(data_dir, exist_ok=True)

        # Index for fast lookups: {normalized_label: entity_id}
        self.labels_index = {}
        self.aliases_index = {}

        # Initialize the index
        self._initialize()

    def _initialize(self):
        """Load existing index or download data and build index if needed."""
        # Check for existing index FIRST
        if os.path.exists(self.index_file):
            print("Loading existing search index...")
            self._load_index()
            return  # Exit early - we have what we need!

        # Only proceed if no index exists
        print("No index found, need to download and build...")

        # Only if no index exists, then check for the big file
        if not os.path.exists(self.labels_file):
            print("Wikidata labels file not found. Downloading...")
            self._download_labels()

        if not os.path.exists(self.index_file):
            print("Building search index from labels...")
            self._build_index()
        else:
            print("Loading existing search index...")
            self._load_index()

    def _download_labels(self):
        """Download the latest Wikidata labels dump."""
        # Try alternative sources for labels-only data
        urls_to_try = [
            # Try a third-party processed labels dump (if available)
            "https://zenodo.org/record/6477509/files/wikidata-labels-en.json.gz",
            # Fall back to full dump
            "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz",
        ]

        print("Trying to download Wikidata labels...")

        success = False
        for i, url in enumerate(urls_to_try):
            try:
                print(f"Trying source {i + 1}: {url}")
                response = requests.head(url, timeout=30)
                if response.status_code == 200:
                    print(f"Found valid URL: {url}")

                    # Download the file
                    response = requests.get(url, stream=True, timeout=30)
                    response.raise_for_status()

                    total_size = int(response.headers.get('content-length', 0))
                    size_gb = total_size / (1024 ** 3) if total_size > 0 else 0

                    if size_gb > 50:
                        print(f"Warning: Large file ({size_gb:.1f}GB) - this will take several hours")

                    with open(self.labels_file, 'wb') as f:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))

                    print(f"Download complete: {self.labels_file}")
                    success = True
                    break

            except requests.exceptions.RequestException as e:
                print(f"Failed: {e}")
                continue

        if not success:
            print("\n" + "=" * 60)
            print("MANUAL DOWNLOAD REQUIRED")
            print("=" * 60)
            print("Automatic download failed. Please manually download:")
            print("1. Go to: https://dumps.wikimedia.org/wikidatawiki/entities/")
            print("2. Find the latest 'wikidata-YYYYMMDD-all.json.gz' file")
            print(f"3. Download it and save as: {self.labels_file}")
            print("4. Then run the script again")
            raise Exception("Could not automatically download Wikidata dump")

    def _build_index(self):
        """Build searchable index from the labels file with memory-efficient streaming."""
        print("Building search index with streaming...")

        labels_count = 0
        aliases_count = 0
        processed_entities = 0
        skipped_lines = 0

        # Process in smaller chunks to manage memory
        batch_labels = {}
        batch_aliases = {}
        batch_size = 10000  # Process in batches to control memory

        try:
            with gzip.open(self.labels_file, 'rt', encoding='utf-8') as f:
                with tqdm(desc="Processing entities", unit="entities") as pbar:
                    for line_num, line in enumerate(f):
                        try:
                            # Skip the opening bracket on line 0
                            if line_num == 0:
                                if line.strip() == '[':
                                    print("DEBUG: Skipping opening bracket")
                                    continue
                                else:
                                    print(f"DEBUG: Unexpected first line: {line[:100]}")

                            # DEBUG: Show first few lines to understand structure
                            if line_num <= 5:
                                print(f"DEBUG Line {line_num}: {line[:200]}...")

                            # Skip empty lines
                            line = line.strip()
                            if not line:
                                continue

                            # Remove trailing comma if present (JSON array format)
                            if line.endswith(','):
                                line = line[:-1]

                            # Skip closing bracket
                            if line.strip() == ']':
                                print("DEBUG: Found closing bracket, finishing")
                                break

                            # Parse JSON for this single line
                            entity_data = json.loads(line)

                            # DEBUG: Show structure of first few entities
                            if line_num <= 5:
                                print(f"DEBUG Line {line_num} - Entity keys: {list(entity_data.keys())}")
                                print(f"DEBUG Line {line_num} - Entity ID: {entity_data.get('id', 'NO_ID')}")

                            entity_id = entity_data.get('id')

                            if not entity_id or not entity_id.startswith('Q'):
                                skipped_lines += 1
                                continue

                            # If we get here, we should increment processed_entities
                            processed_entities += 1

                            if processed_entities <= 3:
                                print(f"DEBUG SUCCESS: Processing entity {entity_id}")

                            # Process labels
                            labels = entity_data.get('labels', {})
                            for lang, label_info in labels.items():
                                if lang in ['en', 'en-gb', 'en-us']:  # Focus on English
                                    label_text = label_info.get('value', '').strip()
                                    if label_text:
                                        normalized = self._normalize_label(label_text)
                                        if normalized and len(normalized) > 1:  # Skip very short labels
                                            batch_labels[normalized] = entity_id
                                            labels_count += 1

                            # Process aliases
                            aliases = entity_data.get('aliases', {})
                            for lang, alias_list in aliases.items():
                                if lang in ['en', 'en-gb', 'en-us']:
                                    for alias_info in alias_list:
                                        alias_text = alias_info.get('value', '').strip()
                                        if alias_text:
                                            normalized = self._normalize_label(alias_text)
                                            if normalized and len(normalized) > 1:  # Skip very short aliases
                                                batch_aliases[normalized] = entity_id
                                                aliases_count += 1

                            # Merge batches periodically to control memory
                            if len(batch_labels) >= batch_size or len(batch_aliases) >= batch_size:
                                self.labels_index.update(batch_labels)
                                self.aliases_index.update(batch_aliases)
                                batch_labels.clear()
                                batch_aliases.clear()

                            pbar.update(1)

                        except json.JSONDecodeError:
                            skipped_lines += 1
                            continue
                        except KeyError:
                            skipped_lines += 1
                            continue
                        except Exception as e:
                            skipped_lines += 1
                            if line_num % 100000 == 0:  # Only print occasional errors
                                print(f"Error processing line {line_num}: {e}")
                            continue

            # Merge any remaining batch data
            if batch_labels:
                self.labels_index.update(batch_labels)
            if batch_aliases:
                self.aliases_index.update(batch_aliases)

            print(f"\nIndex building complete!")
            print(f"Final stats: {processed_entities:,} entities processed")
            print(f"Index built: {labels_count:,} labels, {aliases_count:,} aliases")
            print(f"Skipped lines: {skipped_lines:,}")
            print(f"Final index sizes: {len(self.labels_index):,} labels, {len(self.aliases_index):,} aliases")

        except Exception as e:
            print(f"\nFatal error during index building: {e}")
            print("This might be due to memory constraints or file corruption")
            raise

        self._save_index()

    def _save_index(self):
        """Save the built index to disk."""
        index_data = {
            'labels': self.labels_index,
            'aliases': self.aliases_index
        }

        with open(self.index_file, 'wb') as f:
            pickle.dump(index_data, f)

        print(f"Index saved to {self.index_file}")

    def _load_index(self):
        """Load the pre-built index from disk."""
        with open(self.index_file, 'rb') as f:
            index_data = pickle.load(f)

        self.labels_index = index_data['labels']
        self.aliases_index = index_data['aliases']

        print(f"Loaded index: {len(self.labels_index):,} labels, {len(self.aliases_index):,} aliases")

    def _normalize_label(self, label: str) -> str:
        """
        Normalize a label for consistent matching.

        Args:
            label: Raw label text

        Returns:
            Normalized label for indexing
        """
        if not label:
            return ""

        # Convert to lowercase
        normalized = label.lower().strip()

        # Remove common punctuation but keep meaningful characters
        normalized = re.sub(r'["""''`]', '', normalized)  # Remove quotes
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace

        # Handle URL decoding for entity names that might be URL-encoded
        try:
            normalized = unquote(normalized)
        except:
            pass

        return normalized

    def _generate_search_variants(self, search_term: str) -> Set[str]:
        """
        Generate search variants for better matching.

        Args:
            search_term: Original search term

        Returns:
            Set of search variants to try
        """
        variants = set()
        base_term = self._normalize_label(search_term)
        variants.add(base_term)

        # Handle underscores vs spaces
        variants.add(base_term.replace('_', ' '))
        variants.add(base_term.replace(' ', '_'))

        # Handle common abbreviations
        if '.' in base_term:
            # Try without periods: "U.S.A." -> "USA"
            variants.add(base_term.replace('.', ''))
            # Try with spaces: "U.S.A." -> "U S A"
            variants.add(base_term.replace('.', ' '))

        # Handle initials: "JFK" -> "J.F.K."
        if len(base_term) <= 5 and base_term.isupper():
            variants.add('.'.join(base_term) + '.')

        # Handle "the" prefix
        if base_term.startswith('the '):
            variants.add(base_term[4:])
        else:
            variants.add('the ' + base_term)

        return variants

    def search_entity(self, entity_name: str, entity_type: str = None) -> Optional[str]:
        """
        Search for a Wikidata entity and return its URL.

        Args:
            entity_name: Name of the entity to search for
            entity_type: Type hint (for compatibility, not used in local search)

        Returns:
            Wikidata URL if found, None otherwise
        """
        if not entity_name or not entity_name.strip():
            return None

        # Generate search variants
        search_variants = self._generate_search_variants(entity_name)

        # Try exact matches in labels first
        for variant in search_variants:
            if variant in self.labels_index:
                entity_id = self.labels_index[variant]
                return f"http://www.wikidata.org/entity/{entity_id}"

        # Try aliases if no exact label match
        for variant in search_variants:
            if variant in self.aliases_index:
                entity_id = self.aliases_index[variant]
                return f"http://www.wikidata.org/entity/{entity_id}"

        return None

    def link_entity(self, entity_name: str, entity_type: str = None) -> Optional[dict]:
        """
        Link entity method for compatibility with OntologyMapper.
        """
        wikidata_url = self.search_entity(entity_name, entity_type)

        if wikidata_url:
            return {
                'source': 'wikidata',
                'wikidata_uri': wikidata_url,
                'wikidata_id': wikidata_url.split('/')[-1],
            }

        return None

    def save_cache(self):
        """Compatibility method - no caching needed for local operation."""
        pass

    def save_enhanced_cache(self):
        """Compatibility method - no caching needed for local operation."""
        pass

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the loaded index."""
        return {
            'total_labels': len(self.labels_index),
            'total_aliases': len(self.aliases_index),
            'total_entities': len(set(list(self.labels_index.values()) + list(self.aliases_index.values())))
        }
