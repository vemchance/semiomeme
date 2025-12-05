#!/usr/bin/env python3
"""
Semantic Lineage: A Demonstration of Hybrid Querying
=====================================================
Uses Sankey diagram to show flow:
Instances → Meme Entries → Semantic Context
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import plotly.graph_objects as go
from PIL import Image
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

from config.namespaces import SMO, EX, SCHEMA
from config.config import CORPUS_CONFIG
from corpus_faiss_similarity_queries import SimilaritySearch

OUTPUT_DIR = Path(r"X:\PhD\SemioMeme_Graph\04_Analysis\semantic_lineage")
DATASET_BASE_DIR = Path(r"X:\PhD\Datasets\Memes\KYM\Confirmed Images")


class SemanticLineage:

    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("Loading similarity search...")
        self.searcher = SimilaritySearch(
            vision_index_path=str(CORPUS_CONFIG.VISION_INDEX_PATH),
            text_index_path=str(CORPUS_CONFIG.TEXT_INDEX_PATH),
            vision_model_path=str(CORPUS_CONFIG.VISION_MODEL_PATH),
            text_model_path=str(CORPUS_CONFIG.TEXT_MODEL_PATH),
            rdf_graph_path=str(CORPUS_CONFIG.INSTANCE_GRAPH_PATH),
            dataset_base_dir=str(DATASET_BASE_DIR),
            search_mode='all'
        )

        if self.searcher.vision_index:
            self.searcher.vision_index.nprobe = 32
        if self.searcher.text_index:
            self.searcher.text_index.nprobe = 32

        self.graph = self.searcher.rdf_graph
        print(f"Ready. Graph: {len(self.graph)} triples")

    def query(self, image_path: str = None, text_query: str = None, k: int = 50) -> Dict:
        """Run hybrid query."""
        print(f"\n{'=' * 60}")
        print("SEMANTIC LINEAGE QUERY")
        print(f"{'=' * 60}")

        # Layer 1: Similarity
        print(f"\n[1] SIMILARITY: What looks like this?")
        if image_path and text_query:
            instances = self.searcher.search_multimodal_rrf(image_path, text_query, k=k)
            query_type = 'multimodal'
        elif image_path:
            instances = self.searcher.search_by_image(image_path, k=k)
            query_type = 'vision'
        else:
            instances = self.searcher.search_by_text(text_query, k=k)
            query_type = 'text'

        print(f"    → {len(instances)} similar instances")

        # Layer 2: Bridge
        print(f"\n[2] BRIDGE: What ARE these?")
        meme_entries = defaultdict(list)

        for inst in instances:
            uri = inst.get('instance_uri')
            if uri:
                for parent in self.graph.objects(URIRef(uri), SMO.belongsTo):
                    label = self._label(parent)
                    if label:
                        inst['meme_entry_label'] = label
                        inst['meme_entry_uri'] = str(parent)
                        meme_entries[label].append({
                            'uri': str(parent),
                            'instance': inst
                        })
                        break

        print(f"    → {len(meme_entries)} meme entries")
        for entry, items in sorted(meme_entries.items(), key=lambda x: -len(x[1]))[:5]:
            print(f"       • {entry} ({len(items)} instances)")

        # Layer 3: Semantic - collect all first, then deduplicate
        print(f"\n[3] SEMANTIC: What do they MEAN?")

        # Collect everything first
        raw_semantics = {
            'people': defaultdict(lambda: {'entries': set()}),
            'events': defaultdict(lambda: {'entries': set()}),
            'sites': defaultdict(lambda: {'entries': set()}),
            'subcultures': defaultdict(lambda: {'entries': set()}),
            'series': defaultdict(lambda: {'entries': set()}),
            'meme_refs': defaultdict(lambda: {'entries': set()}),
            'tags': defaultdict(lambda: {'entries': set()}),
        }

        for entry_label, items in meme_entries.items():
            entry_uri = URIRef(items[0]['uri'])
            self._extract_semantics_raw(entry_uri, entry_label, raw_semantics)

        # Deduplicate: higher priority types win
        semantics = self._deduplicate_semantics(raw_semantics)

        print(f"    → {len(semantics['people'])} people")
        print(f"    → {len(semantics['events'])} events")
        print(f"    → {len(semantics['sites'])} sites")
        print(f"    → {len(semantics['series'])} series")
        print(f"    → {len(semantics['meme_refs'])} meme references")
        print(f"    → {len(semantics['tags'])} tags (deduplicated)")

        # Get one representative image per meme entry
        entry_images = {}
        for entry_label, items in meme_entries.items():
            for item in items:
                img_path = item['instance'].get('image_path')
                if img_path and Path(img_path).exists():
                    entry_images[entry_label] = img_path
                    break

        return {
            'query': {'image': image_path, 'text': text_query, 'type': query_type},
            'similarity': {'count': len(instances), 'instances': instances},
            'bridge': {
                'count': len(meme_entries),
                'meme_entries': {k: len(v) for k, v in meme_entries.items()},
                'entry_images': entry_images
            },
            'semantics': {
                'people': {k: {'entries': list(v['entries'])} for k, v in semantics['people'].items()},
                'events': {k: {'entries': list(v['entries'])} for k, v in semantics['events'].items()},
                'sites': {k: {'entries': list(v['entries'])} for k, v in semantics['sites'].items()},
                'subcultures': {k: {'entries': list(v['entries'])} for k, v in semantics['subcultures'].items()},
                'series': {k: {'entries': list(v['entries'])} for k, v in semantics['series'].items()},
                'meme_refs': {k: {'entries': list(v['entries'])} for k, v in semantics['meme_refs'].items()},
                'tags': {k: {'entries': list(v['entries'])} for k, v in semantics['tags'].items()},
            }
        }

    def _extract_semantics_raw(self, uri: URIRef, entry_label: str, raw_semantics: Dict):
        """Extract all semantic attributes without deduplication."""
        for pred, obj in self.graph.predicate_objects(uri):
            obj_label = self._label(obj)
            if not obj_label:
                continue

            if pred == SMO.hasTag:
                raw_semantics['tags'][obj_label.lower()]['entries'].add(entry_label)

            elif pred == SMO.partOfSeries:
                raw_semantics['series'][obj_label]['entries'].add(entry_label)

            elif pred == SMO.mentions:
                types = [str(t).split('/')[-1] for t in self.graph.objects(obj, RDF.type)]

                if 'PersonEntry' in types:
                    raw_semantics['people'][obj_label]['entries'].add(entry_label)
                elif 'EventEntry' in types:
                    raw_semantics['events'][obj_label]['entries'].add(entry_label)
                elif 'SiteEntry' in types:
                    raw_semantics['sites'][obj_label]['entries'].add(entry_label)
                elif 'SubcultureEntry' in types:
                    raw_semantics['subcultures'][obj_label]['entries'].add(entry_label)
                elif 'MemeEntry' in types:
                    raw_semantics['meme_refs'][obj_label]['entries'].add(entry_label)

    def _deduplicate_semantics(self, raw_semantics: Dict) -> Dict:
        """
        Deduplicate entities across categories.
        Priority (highest to lowest):
        PersonEntry > EventEntry > SiteEntry > SubcultureEntry > Series > MemeEntry > Tag
        """
        priority_order = ['people', 'events', 'sites', 'subcultures', 'series', 'meme_refs', 'tags']
        claimed_names = set()

        deduplicated = {
            'people': {},
            'events': {},
            'sites': {},
            'subcultures': {},
            'series': {},
            'meme_refs': {},
            'tags': {},
        }

        for category in priority_order:
            items = raw_semantics[category]

            for name, data in items.items():
                name_lower = name.lower().strip()

                if name_lower not in claimed_names:
                    claimed_names.add(name_lower)
                    deduplicated[category][name] = data

        return deduplicated

    def _label(self, uri) -> Optional[str]:
        for label in self.graph.objects(uri, RDFS.label):
            return str(label)
        return None

    def visualize(self, results: Dict, query_image_path: str = None, output_name: str = "semantic_lineage"):
        """Create Sankey diagram with legend at bottom and filtered tags."""

        meme_entries = results['bridge']['meme_entries']
        sem = results['semantics']
        n_instances = results['similarity']['count']

        # Stopwords to filter out bad tags
        STOPWORDS = {
            'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should',
            'now', 'coming', 'going', 'being', 'having', 'doing', 'made', 'found',
            'said', 'get', 'got', 'has', 'had', 'was', 'were', 'been', 'be', 'are',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
            'what', 'which', 'who', 'whom', 'would', 'could', 'might', 'must', 'shall',
            'may', 'need', 'dare', 'ought', 'used', 'also', 'back', 'even', 'still',
            'well', 'way', 'because', 'if', 'while', 'although', 'though', 'after',
            'new', 'old', 'high', 'low', 'big', 'small', 'large', 'long', 'short',
            'good', 'bad', 'great', 'little', 'much', 'many', 'one', 'two', 'first',
            'last', 'over', 'out', 'up', 'down', 'off', 'about', 'against', 'any',
        }

        MIN_TAG_LENGTH = 3

        def is_valid_tag(tag_name):
            """Filter out stopwords and short tags."""
            tag_lower = tag_name.lower().strip()
            if len(tag_lower) < MIN_TAG_LENGTH:
                return False
            if tag_lower in STOPWORDS:
                return False
            return True

        # Build node list
        nodes = []
        node_colors = []

        # Node 0: Query/Instances
        nodes.append(f"Query ({n_instances} instances)")
        node_colors.append('#3498db')

        # Meme entry nodes
        sorted_entries = sorted(meme_entries.items(), key=lambda x: -x[1])
        entry_indices = {}
        for entry_name, count in sorted_entries:
            entry_indices[entry_name] = len(nodes)
            nodes.append(f"{entry_name[:25]} ({count})")
            node_colors.append('#2ecc71')

        # Category configs
        category_configs = [
            ('people', 'Person', '#e74c3c'),
            ('events', 'Event', '#f39c12'),
            ('sites', 'Site', '#9b59b6'),
            ('series', 'Series', '#1abc9c'),
            ('meme_refs', 'Meme Ref', '#27ae60'),
            ('tags', 'Tag', '#7f8c8d'),
        ]

        semantic_indices = {}
        for cat_key, cat_label, color in category_configs:
            items = sem.get(cat_key, {})
            sorted_items = sorted(items.items(), key=lambda x: -len(x[1]['entries']))

            # Filter tags
            if cat_key == 'tags':
                sorted_items = [(name, data) for name, data in sorted_items if is_valid_tag(name)]

            # Take top 8
            for item_name, item_data in sorted_items[:8]:
                semantic_indices[(cat_key, item_name)] = len(nodes)
                nodes.append(f"{item_name[:22]}")
                node_colors.append(color)

        # Build links
        sources = []
        targets = []
        values = []
        link_colors = []

        # Links: Query → Meme Entries
        for entry_name, count in sorted_entries:
            sources.append(0)
            targets.append(entry_indices[entry_name])
            values.append(count)
            link_colors.append('rgba(52, 152, 219, 0.4)')

        # Links: Meme Entries → Semantic entities
        for cat_key, cat_label, color in category_configs:
            items = sem.get(cat_key, {})
            sorted_items = sorted(items.items(), key=lambda x: -len(x[1]['entries']))

            if cat_key == 'tags':
                sorted_items = [(name, data) for name, data in sorted_items if is_valid_tag(name)]

            hex_color = color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            rgba = f'rgba({r}, {g}, {b}, 0.4)'

            for item_name, item_data in sorted_items[:8]:
                for entry_name in item_data['entries']:
                    if entry_name in entry_indices:
                        sources.append(entry_indices[entry_name])
                        targets.append(semantic_indices[(cat_key, item_name)])
                        values.append(1)
                        link_colors.append(rgba)

        # Create Sankey with LARGER FONTS
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=25,
                thickness=30,
                line=dict(color='white', width=2),
                label=nodes,
                color=node_colors,
                hovertemplate='%{label}<extra></extra>'
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
            ),
            textfont=dict(size=14, color='black', family='Arial Black')
        )])

        # Legend at bottom
        legend_items = [
            ('Query', '#3498db'),
            ('Meme Entry', '#2ecc71'),
            ('Person', '#e74c3c'),
            ('Event', '#f39c12'),
            ('Site', '#9b59b6'),
            ('Series', '#1abc9c'),
            ('Meme Ref', '#27ae60'),
            ('Tag', '#7f8c8d'),
        ]

        # Position legend horizontally at bottom
        annotations = []
        start_x = 0.08
        spacing = 0.11

        for i, (label, color) in enumerate(legend_items):
            x_pos = start_x + i * spacing
            annotations.append(dict(
                x=x_pos, y=-0.06,
                xref='paper', yref='paper',
                text=f'<b style="color:{color}">■</b> {label}',
                showarrow=False,
                font=dict(size=14),
                align='center'
            ))

        fig.update_layout(
            title_text="Semantic Lineage: Hybrid Query Flow",
            title_font_size=24,
            font_size=14,
            width=1600,
            height=1000,
            paper_bgcolor='white',
            margin=dict(b=100, t=80, l=50, r=50),
            annotations=annotations
        )

        # Save HTML
        html_path = OUTPUT_DIR / f"{output_name}_{time.strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(html_path))
        print(f"\nSaved interactive: {html_path}")

        # Save PNG
        png_path = OUTPUT_DIR / f"{output_name}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        fig.write_image(str(png_path), scale=2)
        print(f"Saved static: {png_path}")

        return png_path

    def visualize_with_images(self, results: Dict, query_image_path: str = None, output_name: str = "semantic_lineage"):
        """Combined visualization: example images + Sankey diagram."""

        # Create figure with GridSpec
        fig = plt.figure(figsize=(20, 14), facecolor='white')
        gs = GridSpec(2, 5, height_ratios=[0.2, 0.8], hspace=0.1, wspace=0.08)

        # Row 1: Query + 4 example images
        ax_query = fig.add_subplot(gs[0, 0])
        self._draw_image(ax_query, query_image_path, "QUERY")

        top_entries = sorted(results['bridge']['meme_entries'].items(), key=lambda x: -x[1])[:4]
        entry_images = results['bridge']['entry_images']

        for i, (entry_name, count) in enumerate(top_entries):
            ax = fig.add_subplot(gs[0, i + 1])
            img_path = entry_images.get(entry_name)
            self._draw_image(ax, img_path, f"{entry_name[:18]}\n({count} instances)")

        # Row 2: Sankey placeholder
        ax_sankey = fig.add_subplot(gs[1, :])
        ax_sankey.axis('off')
        ax_sankey.text(0.5, 0.5, "See Sankey diagram in separate file",
                       ha='center', va='center', fontsize=18, style='italic')

        plt.tight_layout()

        # Save matplotlib figure
        combined_path = OUTPUT_DIR / f"{output_name}_combined_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(combined_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        # Generate Sankey separately
        sankey_path = self.visualize(results, query_image_path, output_name)

        print(f"Saved combined: {combined_path}")
        return sankey_path

    def _draw_image(self, ax, image_path: str, title: str):
        """Draw an image with title."""
        if image_path and Path(image_path).exists():
            img = Image.open(image_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "—", ha='center', va='center', fontsize=16)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    def export_json(self, results: Dict, output_name: str = "semantic_lineage"):
        """Export results to JSON."""
        output_path = OUTPUT_DIR / f"{output_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved: {output_path}")
        return output_path


def main():
    QUERY_IMAGE = r"X:\Downloads\kek.png"
    QUERY_TEXT = None
    K = 50

    sl = SemanticLineage()
    results = sl.query(image_path=QUERY_IMAGE, text_query=QUERY_TEXT, k=K)

    sl.visualize(results, query_image_path=QUERY_IMAGE)
    sl.export_json(results)


if __name__ == '__main__':
    main()