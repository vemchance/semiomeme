"""
Phase 1: Extract REBEL relations from OCR text and save to JSON
No graph operations
"""

import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import torch
import gc

# Import REBEL extractor
from kgraph.builder.rebel_extractor import REBELTripleExtractor

from config.config import RETRIEVAL_CONFIG, CORPUS_CONFIG

TEXT_CSV_FILES = CORPUS_CONFIG.TEXT_CSV_FILES
OUTPUT_DIR = CORPUS_CONFIG.OUTPUT_DIR
OUTPUT_FILE = OUTPUT_DIR / "rebel_extractions.json"

class REBELExtractor:
    """Extract REBEL relations from OCR text and save to file"""

    def __init__(self, output_file="rebel_extractions.json"):
        self.output_file = output_file
        self.text_data = {}

        # Initialize REBEL
        print("Initializing REBEL extractor...")
        self.rebel_extractor = REBELTripleExtractor(
            model_name="Babelscape/rebel-large"
        )

        # SPEED OPTIMIZATIONS
        self.rebel_extractor.default_batch_size = 128  # GPU batch size
        self.rebel_extractor.beam_width = 1  # Greedy decoding - 3x faster than beam_width=3
        self.rebel_extractor.max_seq_length = 512  # Reduced from 768 - 20% faster

        print(f"GPU batch size: {self.rebel_extractor.default_batch_size}")
        print(f"Beam width: {self.rebel_extractor.beam_width} (greedy decoding for speed)")
        print(f"Max sequence length: {self.rebel_extractor.max_seq_length}")

    def load_ocr_data(self):
        """Load OCR text from CSV files - CONFIRMED ONLY"""
        print("\n=== Loading OCR Data ===")

        for csv_file in TEXT_CSV_FILES:
            if not Path(csv_file).exists():
                print(f"Warning: CSV file not found: {csv_file}")
                continue

            # Skip unconfirmed memes
            csv_filename = Path(csv_file).name.lower()
            if 'unconfirmed' in csv_filename:
                print(f"Skipping unconfirmed: {Path(csv_file).name}")
                continue

            try:
                print(f"Loading: {Path(csv_file).name}")
                df = pd.read_csv(csv_file)

                # Find filename column
                if 'Image Ref' in df.columns:
                    filename_column = 'Image Ref'
                elif 'file' in df.columns:
                    filename_column = 'file'
                else:
                    print(f"  Warning: No filename column found")
                    continue

                # Check for text column
                if 'Text' not in df.columns:
                    print(f"  Warning: No 'Text' column found")
                    continue

                # Process each row with aggressive filtering
                loaded_count = 0
                skipped_short = 0
                skipped_numbers = 0
                skipped_few_words = 0
                skipped_too_long = 0

                for _, row in df.iterrows():
                    # Extract filename
                    full_path = row[filename_column]
                    filename = full_path.split('\\')[-1] if '\\' in full_path else Path(full_path).name

                    # Get text
                    text = row['Text']
                    if pd.notna(text) and str(text).strip():
                        text = str(text).strip()

                        # FILTER 1: Skip very short text (< 20 chars)
                        if len(text) < 20:
                            skipped_short += 1
                            continue

                        # FILTER 2: Skip if only numbers/dates
                        if text.replace(' ', '').replace('-', '').replace('/', '').replace(':', '').replace('.',
                                                                                                            '').isdigit():
                            skipped_numbers += 1
                            continue

                        # FILTER 3: Skip if less than 3 words
                        word_count = len(text.split())
                        if word_count < 3:
                            skipped_few_words += 1
                            continue

                        # FILTER 4: Skip very long texts (> 512 words) since we're not chunking
                        if word_count > 512:
                            skipped_too_long += 1
                            continue

                        # Passed all filters
                        self.text_data[filename] = text
                        loaded_count += 1

                print(f"  Loaded {loaded_count} text entries")
                print(
                    f"  Filtered: {skipped_short} too short, {skipped_numbers} numbers-only, {skipped_few_words} < 3 words, {skipped_too_long} > 512 words")

            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")

        print(f"\nTotal OCR entries loaded: {len(self.text_data)}")
        print("Filters applied: 20+ chars, 3+ words, no numbers-only text")

    def extract_relations(self):
        """Extract REBEL relations from all OCR texts"""
        print(f"\n=== Extracting REBEL Relations ===")
        print(f"Processing {len(self.text_data)} OCR texts")

        # Prepare data
        filenames = list(self.text_data.keys())
        texts = list(self.text_data.values())

        # Storage for results
        results = {}

        # Process in chunks for progress visibility
        chunk_size = 5000
        num_chunks = (len(texts) + chunk_size - 1) // chunk_size

        print(f"Processing in {num_chunks} chunks of {chunk_size} texts")

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(texts))

            print(f"\n=== Chunk {chunk_idx + 1}/{num_chunks}: Processing texts {start_idx}-{end_idx} ===")

            chunk_texts = texts[start_idx:end_idx]
            chunk_filenames = filenames[start_idx:end_idx]

            # Extract relations for this chunk
            print("Extracting relations...")
            # Use direct batch processing - NO text chunking
            chunk_relations = self.rebel_extractor.extract_triples_batch(
                chunk_texts,
                batch_size=None,  # Uses default_batch_size (128)
                show_progress=True
            )

            # Store results
            for filename, relations in zip(chunk_filenames, chunk_relations):
                if relations:
                    results[filename] = relations

            print(f"Chunk {chunk_idx + 1} complete: {len([r for r in chunk_relations if r])} texts with relations")

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        return results

    def save_results(self, results):
        """Save extraction results to JSON"""
        print(f"\n=== Saving Results ===")

        # Convert to serializable format
        output_data = {
            'total_texts': len(self.text_data),
            'texts_with_relations': len(results),
            'extractions': {}
        }

        for filename, relations in results.items():
            # Store as list of [subject, relation, object] triples
            output_data['extractions'][filename] = [
                [subj, rel, obj] for subj, rel, obj in relations
            ]

        # Save to JSON
        output_path = Path(self.output_file)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved to: {output_path}")
        print(f"Total texts processed: {output_data['total_texts']}")
        print(f"Texts with relations: {output_data['texts_with_relations']}")

        # Calculate statistics
        total_relations = sum(len(rels) for rels in results.values())
        print(f"Total relations extracted: {total_relations}")

        if results:
            avg_relations = total_relations / len(results)
            print(f"Average relations per text: {avg_relations:.2f}")


if __name__ == "__main__":
    # Initialize extractor
    extractor = REBELExtractor(output_file=OUTPUT_FILE)

    # Load OCR data
    extractor.load_ocr_data()

    if not extractor.text_data:
        print("No OCR data loaded. Exiting.")
        exit(1)

    # Extract relations
    results = extractor.extract_relations()

    # Save results
    extractor.save_results(results)

    print("\n=== Extraction Complete ===")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("Next step: Run add_rebel_to_graph.py to add relations to graph")