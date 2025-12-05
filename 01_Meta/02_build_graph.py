#!/usr/bin/env python3
"""Graph building CLI"""

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config.config import META_CONFIG
from kgraph.builder.pipeline import MemeGraphPipeline
from kgraph.utils import apply_pattern_based_corrections


def main():
    pipeline = MemeGraphPipeline()
    graph, entity_registry = pipeline.process_meme_data(
        str(META_CONFIG.INPUT_FILE),
        batch_size=META_CONFIG.GRAPH_BUILD['batch_size'],
        test_mode=META_CONFIG.GRAPH_BUILD['test_mode'],
        sample_size=META_CONFIG.GRAPH_BUILD['sample_size'],
        cache_dir=str(META_CONFIG.CACHE_DIR),
        enable_rebel=META_CONFIG.GRAPH_BUILD['enable_rebel'],
        use_local_wikidata=META_CONFIG.GRAPH_BUILD['use_local_wikidata'],
        enrich_rebel=META_CONFIG.GRAPH_BUILD['enrich_rebel']
    )
    # Apply corrections and save
    apply_pattern_based_corrections(graph, entity_registry)
    output_file = META_CONFIG.DEFAULT_GRAPH

    # CREATE THE DIRECTORY IF IT DOESN'T EXIST
    output_file.parent.mkdir(parents=True, exist_ok=True)

    graph.serialize(destination=str(output_file), format="turtle")
    print(f"Graph saved to: {output_file}")

if __name__ == "__main__":
    main()