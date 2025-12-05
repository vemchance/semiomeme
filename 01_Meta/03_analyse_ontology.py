#!/usr/bin/env python3
"""Ontology analysis CLI entry point"""
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config.config import META_CONFIG
from kgraph.analysis.ontology_analyser import OntologyAnalyser


def main():
    graph_path = str(META_CONFIG.DEFAULT_GRAPH)

    print(f"Analysing graph: {graph_path}")

    # Create and run analyser
    analyser = OntologyAnalyser(
        str(graph_path),
        str(META_CONFIG.ANALYSIS['output_dir'])
    )

    analyser.run_all(
        generate_stats=META_CONFIG.ANALYSIS['generate_statistics'],
        analyse_connectivity=META_CONFIG.ANALYSIS['analyse_connectivity']
    )

    print(f"Analysis complete! Results in: {analyser.output_dir}")


if __name__ == "__main__":
    main()