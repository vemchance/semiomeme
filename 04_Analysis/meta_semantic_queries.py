"""
Semantic Query Example: Entity Analysis in Meme Culture
Usage: PersonEntry queries
"""

import rdflib
from rdflib import Graph, Namespace, RDF, RDFS, OWL
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import pandas as pd
import os
import networkx as nx
import numpy as np
from datetime import datetime
import seaborn as sns
from adjustText import adjust_text


from config.namespaces import SMO, EX, SCHEMA, OWL
from config.config import META_CONFIG, CORPUS_CONFIG

meta_graph = META_CONFIG.DEFAULT_GRAPH  # path to meta only graph
corpus_graph = CORPUS_CONFIG.INSTANCE_GRAPH_PATH  # path to meta + corpus graph
instance_graph = CORPUS_CONFIG.INSTANCE_ONLY_GRAPH_PATH  # instance only graph path

# ============================================================================
# FILE PATHS
# ============================================================================
META_GRAPH_PATH = str(meta_graph)
OUTPUT_FOLDER = r"X:\PhD\SemioMeme_Graph\04_Analysis\meta_semantic_queries"

# Target entity to analyze
TARGET_ENTITY = "elon_musk"

# ============================================================================
# SETUP
# ============================================================================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
if not OUTPUT_FOLDER.endswith(os.sep):
    OUTPUT_FOLDER += os.sep

# Professional visualization style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# Load graph
print("Loading meta graph...")
g = Graph()
g.parse(META_GRAPH_PATH, format="turtle")
print(f"Loaded {len(g)} triples")

g.bind("smo", SMO)
g.bind("ex", EX)
g.bind("schema", SCHEMA)
g.bind("owl", OWL)

# ============================================================================
# TARGET ENTITY SETUP
# ============================================================================
entity_uri = EX[TARGET_ENTITY]
print(f"\nAnalyzing entity: {entity_uri}")

entity_type_query = f"""
SELECT ?type ?label
WHERE {{
    <{entity_uri}> a ?type ;
                   rdfs:label ?label .
    FILTER(?type IN (smo:PersonEntry, smo:MemeEntry, smo:EventEntry, smo:SiteEntry, smo:SubcultureEntry))
}}
LIMIT 1
"""

entity_info = list(g.query(entity_type_query, initNs={"smo": SMO, "rdfs": RDFS}))
if not entity_info:
    print(f"ERROR: Entity {entity_uri} not found in graph!")
    exit(1)

entity_type = str(entity_info[0][0]).split('/')[-1]
entity_label = str(entity_info[0][1])

print(f"Type: {entity_type}")
print(f"Label: {entity_label}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def normalize_label(label):
    """Normalize labels to lowercase for case-insensitive comparison"""
    return str(label).strip().lower()

def get_color_for_type(entity_type):
    """Consistent color mapping for entity types"""
    color_map = {
        'target': '#E63946',       # Bright red for target
        'MemeEntry': '#457B9D',     # Blue for memes
        'meme': '#457B9D',
        'PersonEntry': '#52B788',   # Green for people
        'EventEntry': '#F1C453',    # Yellow for events
        'SiteEntry': '#F4A261',     # Orange for sites
        'SubcultureEntry': '#9B5DE5',  # Purple for subcultures
        'default': '#A8DADC'       # Light blue for unknown
    }
    return color_map.get(entity_type, color_map['default'])

# ============================================================================
# 1. QUERY AND PROCESS DATA (WITH CASE-INSENSITIVE MERGING)
# ============================================================================
print("\n" + "=" * 80)
print(f"ANALYZING: {entity_label.upper()}")
print("=" * 80)

# Query all connections - WITHOUT rdfs:label to avoid duplicates
connections_query = f"""
SELECT DISTINCT ?meme ?connectionType ?year ?viewCount ?commentCount
WHERE {{
    {{
        ?meme a smo:MemeEntry ;
              smo:mentions <{entity_uri}> .
        BIND("mentions" as ?connectionType)
    }} UNION {{
        ?meme a smo:MemeEntry ;
              smo:hasTag <{entity_uri}> .
        BIND("hasTag" as ?connectionType)
    }} UNION {{
        ?meme a smo:MemeEntry ;
              smo:partOfSeries <{entity_uri}> .
        BIND("partOfSeries" as ?connectionType)
    }}
    OPTIONAL {{ ?meme smo:yearCreated ?year }}
    OPTIONAL {{ ?meme smo:viewCount ?viewCount }}
    OPTIONAL {{ ?meme smo:commentCount ?commentCount }}
}}
"""

meme_results = list(g.query(connections_query, initNs={"smo": SMO, "rdfs": RDFS}))
print(f"Found {len(meme_results)} meme connections")

# Process with URI-based labels - TRACK ALL CONNECTION TYPES
meme_data = []
meme_data_by_uri = {}  # Track memes by URI with all their connection types
connection_breakdown = defaultdict(list)
temporal_data = defaultdict(lambda: defaultdict(int))

# Connection type priority (higher = more important)
CONNECTION_PRIORITY = {
    'partOfSeries': 3,
    'hasTag': 2,
    'mentions': 1
}

for row in meme_results:
    meme_uri = str(row[0])
    connection_type = str(row[1])
    year = str(row[2]) if row[2] else None
    views = int(row[3]) if row[3] else 0
    comments = int(row[4]) if row[4] else 0

    # If we've seen this meme, check if new connection type has higher priority
    if meme_uri in meme_data_by_uri:
        existing = meme_data_by_uri[meme_uri]
        existing_priority = CONNECTION_PRIORITY.get(existing['connection'], 0)
        new_priority = CONNECTION_PRIORITY.get(connection_type, 0)

        if new_priority > existing_priority:
            # Update to higher priority connection type
            existing['connection'] = connection_type
        continue

    # Extract label from URI suffix
    meme_label = meme_uri.split('/')[-1].replace('_', ' ').title()

    meme_info = {
        'uri': meme_uri,
        'label': meme_label,
        'connection': connection_type,
        'year': year,
        'views': views,
        'comments': comments,
        'engagement': views + (comments * 10)
    }

    meme_data.append(meme_info)
    meme_data_by_uri[meme_uri] = meme_info
    connection_breakdown[connection_type].append(meme_info)

    if year and year != 'None':
        temporal_data[year][connection_type] += 1

# ============================================================================
# 2. BUILD SECONDARY CONNECTIONS WITH CASE-INSENSITIVE MERGING
# ============================================================================
print("\nProcessing secondary connections with case-insensitive merging...")

# Track all connected entities with case-insensitive merging
entity_connections = defaultdict(lambda: {
    'labels': [],  # Keep track of all label variations
    'count': 0,
    'memes': set(),
    'types': set(),
    'connection_paths': defaultdict(int),
    'total_engagement': 0,
    'display_label': None
})

# Process top memes
top_memes = sorted(meme_data, key=lambda x: x['engagement'], reverse=True)[:30]

for meme in top_memes:
    # Find secondary connections
    secondary_query = f"""
    SELECT DISTINCT ?entity ?type ?property
    WHERE {{
        {{
            <{meme['uri']}> smo:hasTag ?entity .
            BIND("hasTag" as ?property)
        }} UNION {{
            <{meme['uri']}> smo:mentions ?entity .
            BIND("mentions" as ?property)
        }} UNION {{
            <{meme['uri']}> smo:partOfSeries ?entity .
            BIND("partOfSeries" as ?property)
        }}
        ?entity a ?type .
        FILTER(?type IN (smo:PersonEntry, smo:EventEntry, smo:SiteEntry, smo:SubcultureEntry))
        FILTER(?entity != <{entity_uri}>)
        # Exclude any MemeEntry even if it has multiple types
        FILTER NOT EXISTS {{ ?entity a smo:MemeEntry }}
    }}
    """

    for row in g.query(secondary_query, initNs={"smo": SMO, "rdfs": RDFS}):
        entity_uri_secondary = str(row[0])
        entity_type_secondary = str(row[1]).split('/')[-1]
        property_type = str(row[2])

        # Use URI suffix as label (ignore rdfs:label completely)
        entity_label_secondary = entity_uri_secondary.split('/')[-1].replace('_', ' ').title()

        # Normalize for case-insensitive merging
        normalized = normalize_label(entity_label_secondary)

        # Track all label variations
        entity_connections[normalized]['labels'].append(entity_label_secondary)
        entity_connections[normalized]['count'] += 1
        entity_connections[normalized]['memes'].add(meme['label'])
        entity_connections[normalized]['types'].add(entity_type_secondary)
        entity_connections[normalized]['connection_paths'][property_type] += 1
        entity_connections[normalized]['total_engagement'] += meme['engagement']

# Choose the best display label for each normalized entity
for normalized, data in entity_connections.items():
    # Find the best label - prefer one with capitals
    best_label = normalized.title()  # Default fallback
    for label in data['labels']:
        if any(c.isupper() for c in label):  # Has capitals
            best_label = label
            break
    data['display_label'] = best_label

# Get top connected entities
top_secondary = sorted(entity_connections.items(),
                      key=lambda x: x[1]['total_engagement'],
                      reverse=True)[:20]

# ============================================================================
# 3. QUERY ENTITY DETAILS FOR PANEL
# ============================================================================
print("\nQuerying entity RDF details...")

entity_details_query = f"""
SELECT ?property ?value
WHERE {{
    <{entity_uri}> ?property ?value .
}}
ORDER BY ?property
"""

entity_details = list(g.query(entity_details_query))

# Organize entity details
entity_rdf_info = {
    'types': [],
    'labels': [],
    'stats': {},
    'metadata': {}
}

for prop, value in entity_details:
    prop_str = str(prop)
    value_str = str(value)

    if prop == RDF.type:
        entity_rdf_info['types'].append(value_str.split('/')[-1])
    elif prop == RDFS.label:
        entity_rdf_info['labels'].append(value_str)
    elif 'viewCount' in prop_str:
        entity_rdf_info['stats']['Views'] = f"{int(value_str):,}"
    elif 'commentCount' in prop_str:
        entity_rdf_info['stats']['Comments'] = f"{int(value_str):,}"
    elif 'kymID' in prop_str:
        entity_rdf_info['metadata']['KYM ID'] = value_str

# Get connection counts
meme_stats_query = f"""
SELECT 
    (COUNT(DISTINCT ?m1) as ?mentions)
    (COUNT(DISTINCT ?m2) as ?tags)  
    (COUNT(DISTINCT ?m3) as ?series)
WHERE {{
    OPTIONAL {{ ?m1 smo:mentions <{entity_uri}> }}
    OPTIONAL {{ ?m2 smo:hasTag <{entity_uri}> }}
    OPTIONAL {{ ?m3 smo:partOfSeries <{entity_uri}> }}
}}
"""

stats_results = list(g.query(meme_stats_query, initNs={"smo": SMO}))
mentions_count = int(stats_results[0][0]) if stats_results[0][0] else 0
tags_count = int(stats_results[0][1]) if stats_results[0][1] else 0
series_count = int(stats_results[0][2]) if stats_results[0][2] else 0

# ============================================================================
# 4. STANDALONE ENTITY ENTRY DETAILS (PAPER-READY)
# ============================================================================
print("\nCreating standalone entity entry details for paper...")

fig, ax = plt.subplots(figsize=(8, 10))
ax.axis('off')

# Background color
fig.patch.set_facecolor('white')

# Title section with colored background
title_bg = FancyBboxPatch((0.05, 0.88), 0.9, 0.1,
                          boxstyle="round,pad=0.01",
                          facecolor=get_color_for_type(entity_type),
                          alpha=0.15,
                          transform=ax.transAxes)
ax.add_patch(title_bg)

# Main title
ax.text(0.5, 0.94, f'{entity_label}',
        fontsize=18, fontweight='bold', ha='center',
        color=get_color_for_type(entity_type),
        transform=ax.transAxes)

# Subtitle
ax.text(0.5, 0.90, f'smo:{entity_type}',
        fontsize=12, ha='center', style='italic',
        family='monospace', color='#555555',
        transform=ax.transAxes)

# Main content area
y_pos = 0.84

# Section: URI and Basic Info
section_bg = FancyBboxPatch((0.08, y_pos - 0.08), 0.84, 0.075,
                           boxstyle="round,pad=0.01",
                           facecolor='#F0F0F0', alpha=0.5,
                           transform=ax.transAxes)
ax.add_patch(section_bg)

ax.text(0.1, y_pos, 'RESOURCE IDENTIFIER',
        fontsize=10, fontweight='bold', color='#2C3E50',
        transform=ax.transAxes)
y_pos -= 0.03

ax.text(0.12, y_pos, 'URI:', fontsize=10, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.25, y_pos, f'ex:{TARGET_ENTITY}',
        fontsize=10, family='monospace', color='#34495E',
        transform=ax.transAxes)
y_pos -= 0.025

ax.text(0.12, y_pos, 'Graph:', fontsize=10, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.25, y_pos, 'SemioMeme Meta Layer',
        fontsize=10, color='#34495E',
        transform=ax.transAxes)
y_pos -= 0.05

# Section: RDF Types
if entity_rdf_info['types']:
    section_bg = FancyBboxPatch((0.08, y_pos - 0.08), 0.84, 0.075,
                               boxstyle="round,pad=0.01",
                               facecolor='#F0F0F0', alpha=0.5,
                               transform=ax.transAxes)
    ax.add_patch(section_bg)

    ax.text(0.1, y_pos, 'RDF TYPES',
            fontsize=10, fontweight='bold', color='#2C3E50',
            transform=ax.transAxes)
    y_pos -= 0.03

    for i, rdf_type in enumerate(entity_rdf_info['types'][:4]):
        ax.text(0.12, y_pos, f'• smo:{rdf_type}',
                fontsize=9, family='monospace', color='#555555',
                transform=ax.transAxes)
        y_pos -= 0.02
    y_pos -= 0.03

# Section: Labels
if len(entity_rdf_info['labels']) > 1:
    section_bg = FancyBboxPatch((0.08, y_pos - 0.06), 0.84, 0.055,
                               boxstyle="round,pad=0.01",
                               facecolor='#F0F0F0', alpha=0.5,
                               transform=ax.transAxes)
    ax.add_patch(section_bg)

    ax.text(0.1, y_pos, 'LABELS',
            fontsize=10, fontweight='bold', color='#2C3E50',
            transform=ax.transAxes)
    y_pos -= 0.03

    for label in entity_rdf_info['labels'][:3]:
        ax.text(0.12, y_pos, f'"{label}"',
                fontsize=9, color='#555555', style='italic',
                transform=ax.transAxes)
        y_pos -= 0.02
    y_pos -= 0.03

# Section: Engagement Metrics
if entity_rdf_info['stats']:
    section_bg = FancyBboxPatch((0.08, y_pos - 0.09), 0.84, 0.085,
                               boxstyle="round,pad=0.01",
                               facecolor='#FFE5E5', alpha=0.5,
                               transform=ax.transAxes)
    ax.add_patch(section_bg)

    ax.text(0.1, y_pos, 'ENGAGEMENT METRICS',
            fontsize=10, fontweight='bold', color='#2C3E50',
            transform=ax.transAxes)
    y_pos -= 0.03

    # Create two columns for stats
    stats_items = list(entity_rdf_info['stats'].items())
    for i, (stat, value) in enumerate(stats_items):
        x_offset = 0.12 if i % 2 == 0 else 0.52
        y_offset = y_pos - (i // 2) * 0.025

        ax.text(x_offset, y_offset, f'{stat}:',
                fontsize=9, transform=ax.transAxes)
        ax.text(x_offset + 0.15, y_offset, value,
                fontsize=9, fontweight='bold', color='#E74C3C',
                transform=ax.transAxes)

    y_pos -= ((len(stats_items) + 1) // 2) * 0.025 + 0.04

# Section: Meme Connections
section_bg = FancyBboxPatch((0.08, y_pos - 0.12), 0.84, 0.115,
                           boxstyle="round,pad=0.01",
                           facecolor='#E5F3FF', alpha=0.5,
                           transform=ax.transAxes)
ax.add_patch(section_bg)

ax.text(0.1, y_pos, 'MEME NETWORK CONNECTIONS',
        fontsize=10, fontweight='bold', color='#2C3E50',
        transform=ax.transAxes)
y_pos -= 0.03

# Connection stats in a grid
conn_stats = [
    ('Mentioned in:', f'{mentions_count} memes'),
    ('Used as tag:', f'{tags_count} memes'),
    ('Part of series:', f'{series_count} memes')
]

for i, (label, value) in enumerate(conn_stats):
    ax.text(0.12, y_pos, label, fontsize=9,
            transform=ax.transAxes)
    ax.text(0.35, y_pos, value, fontsize=9,
            fontweight='bold', color='#3498DB',
            transform=ax.transAxes)
    y_pos -= 0.025

# Total connections
ax.text(0.12, y_pos - 0.01, 'TOTAL CONNECTIONS:',
        fontsize=10, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.35, y_pos - 0.01, f'{len(meme_data)} memes',
        fontsize=10, fontweight='bold', color='#2980B9',
        transform=ax.transAxes)
y_pos -= 0.05

# Section: Metadata
if entity_rdf_info['metadata']:
    section_bg = FancyBboxPatch((0.08, y_pos - 0.08), 0.84, 0.075,
                               boxstyle="round,pad=0.01",
                               facecolor='#F0F0F0', alpha=0.5,
                               transform=ax.transAxes)
    ax.add_patch(section_bg)

    ax.text(0.1, y_pos, 'METADATA',
            fontsize=10, fontweight='bold', color='#2C3E50',
            transform=ax.transAxes)
    y_pos -= 0.03

    for key, value in list(entity_rdf_info['metadata'].items())[:3]:
        ax.text(0.12, y_pos, f'{key}:', fontsize=9,
                transform=ax.transAxes)
        ax.text(0.35, y_pos, str(value)[:30],
                fontsize=9, family='monospace', color='#555555',
                transform=ax.transAxes)
        y_pos -= 0.02

# Footer with SPARQL example
footer_bg = FancyBboxPatch((0.08, 0.05), 0.84, 0.12,
                          boxstyle="round,pad=0.01",
                          facecolor='#2C3E50', alpha=0.1,
                          transform=ax.transAxes)
ax.add_patch(footer_bg)

ax.text(0.1, 0.15, 'SPARQL QUERY EXAMPLE:',
        fontsize=9, fontweight='bold', color='#2C3E50',
        transform=ax.transAxes)

query_text = f"""SELECT ?meme ?label WHERE {{
  ?meme smo:mentions ex:{TARGET_ENTITY} ;
        rdfs:label ?label .
}} LIMIT 5"""

ax.text(0.1, 0.07, query_text,
        fontsize=7, family='monospace', color='#34495E',
        transform=ax.transAxes)

# Main border
main_border = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                            boxstyle="round,pad=0.02",
                            linewidth=2, edgecolor='#2C3E50',
                            facecolor='none',
                            transform=ax.transAxes)
ax.add_patch(main_border)

# Watermark
ax.text(0.98, 0.01, 'SemioMeme Knowledge Graph',
        fontsize=7, ha='right', style='italic', alpha=0.4,
        transform=ax.transAxes)

plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + f"{TARGET_ENTITY}_entity_entry_details.png",
           bbox_inches='tight', dpi=300, facecolor='white')
print(f"Saved: {OUTPUT_FOLDER}{TARGET_ENTITY}_entity_entry_details.png")
plt.close()

# ============================================================================
# 5. STANDALONE NETWORK VISUALIZATION
# ============================================================================
print("\nCreating standalone network visualization...")

fig, ax = plt.subplots(figsize=(14, 10))

# Build network
G = nx.Graph()
G.add_node(entity_label, type='target', layer=0, engagement=1000000)

# 25 memes
top_memes_for_viz = sorted(meme_data, key=lambda x: x['engagement'], reverse=True)[:20]

for meme in top_memes_for_viz:
    meme_label = meme['label'][:15] + '...' if len(meme['label']) > 15 else meme['label']
    G.add_node(meme_label,
               type='meme',
               layer=1,
               connection=meme['connection'],
               engagement=meme['engagement'])
    G.add_edge(entity_label, meme_label,
               weight=meme['engagement']/1000000,
               connection=meme['connection'])

# 15 entities
added_labels = set()
for normalized, data in top_secondary[:20]:
    display_label = data['display_label']
    if len(display_label) > 15:
        display_label = display_label[:15] + '...'
    entity_type_sec = list(data['types'])[0]

    if display_label in added_labels or display_label in G.nodes():
        continue

    G.add_node(display_label,
               type=entity_type_sec,
               layer=2,
               engagement=data['total_engagement'])
    added_labels.add(display_label)

    for meme_label in data['memes']:
        for node in G.nodes():
            if G.nodes[node].get('layer') == 1:
                truncated = meme_label[:15] + '...' if len(meme_label) > 15 else meme_label
                if (truncated == node or
                    meme_label.startswith(node.replace('...', '')) or
                    node.startswith(meme_label[:27])):
                    G.add_edge(node, display_label, weight=0.3, connection='secondary')
                    break

    if len(added_labels) >= 12:
        break

# Layout
pos = {}
pos[entity_label] = np.array([0, 0])

meme_nodes = [n for n in G.nodes() if G.nodes[n].get('layer') == 1]
entity_nodes = [n for n in G.nodes() if G.nodes[n].get('layer') == 2]

# Memes - 2-level stagger
meme_angle_step = 2 * np.pi / len(meme_nodes)
for i, node in enumerate(meme_nodes):
    angle = i * meme_angle_step
    radius = 0.8 + (0.15 * (i % 2))
    pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

# Entities - 3-level stagger at angle of connected memes
entity_positions = {}
for entity in entity_nodes:
    connected_memes = [n for n in G.neighbors(entity) if G.nodes[n].get('layer') == 1]
    if connected_memes:
        angles = [np.arctan2(pos[m][1], pos[m][0]) for m in connected_memes]
        avg_angle = np.angle(np.mean(np.exp(1j * np.array(angles))))
        entity_positions[entity] = avg_angle
    else:
        entity_positions[entity] = np.random.random() * 2 * np.pi

sorted_entities = sorted(entity_positions.items(), key=lambda x: x[1])

for i, (entity, target_angle) in enumerate(sorted_entities):
    radius = 1.2 + (0.2 * (i % 3))
    pos[entity] = np.array([radius * np.cos(target_angle), radius * np.sin(target_angle)])

# Draw edges
secondary_edges = [(u, v) for u, v in G.edges() if G[u][v].get('connection') == 'secondary']
primary_edges = [(u, v) for u, v in G.edges() if G[u][v].get('connection') != 'secondary']

if secondary_edges:
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=secondary_edges,
                          width=1.5, edge_color='#8E8E93',
                          alpha=0.4, style='dashed')

for u, v in primary_edges:
    connection = G[u][v].get('connection', '')
    if connection == 'mentions':
        color = '#FF6B6B'
    elif connection == 'hasTag':
        color = '#4ECDC4'
    else:
        color = '#95E77E'
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)],
                          width=3, edge_color=color, alpha=0.7)

# Draw nodes
node_colors = []
node_sizes = []

for node in G.nodes():
    node_data = G.nodes[node]
    node_type = node_data.get('type')
    engagement = node_data.get('engagement', 10000)

    node_colors.append(get_color_for_type(node_type))

    if node_type == 'target':
        node_sizes.append(1500)
    elif node_type == 'meme':
        size = 300 + min(700, np.log10(engagement + 1) * 100)
        node_sizes.append(size)
    else:
        size = 400 + min(600, np.log10(engagement + 1) * 80)
        node_sizes.append(size)

nx.draw_networkx_nodes(G, pos, ax=ax,
                      node_color=node_colors,
                      node_size=node_sizes,
                      alpha=0.85,
                      edgecolors='white',
                      linewidths=2)

# Labels with white bbox
for node in G.nodes():
    node_data = G.nodes[node]
    x, y = pos[node]
    label = node

    if node_data.get('type') == 'target':
        ax.text(x, y, label, fontsize=14, fontweight='bold',
               ha='center', va='center', color='black', zorder=100)
    else:
        fontsize = 10 if node_data.get('layer') == 1 else 11
        ax.text(x, y, label, fontsize=fontsize,
               ha='center', va='center', color='black', zorder=100,
               bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                        edgecolor='none', alpha=0.85))

# Legend
legend_elements = [
    plt.scatter([], [], c=get_color_for_type('target'), s=200, label='Target Entity', edgecolors='white', linewidths=2),
    plt.scatter([], [], c=get_color_for_type('MemeEntry'), s=150, label='Meme Entry', edgecolors='white', linewidths=2),
    plt.scatter([], [], c=get_color_for_type('PersonEntry'), s=150, label='Person', edgecolors='white', linewidths=2),
    plt.scatter([], [], c=get_color_for_type('EventEntry'), s=150, label='Event', edgecolors='white', linewidths=2),
    plt.scatter([], [], c=get_color_for_type('SiteEntry'), s=150, label='Site', edgecolors='white', linewidths=2),
    plt.scatter([], [], c=get_color_for_type('SubcultureEntry'), s=150, label='Subculture', edgecolors='white', linewidths=2),
    plt.Line2D([0], [0], color='#FF6B6B', linewidth=3, label='Mentions', alpha=0.7),
    plt.Line2D([0], [0], color='#4ECDC4', linewidth=3, label='Has Tag', alpha=0.7),
    plt.Line2D([0], [0], color='#95E77E', linewidth=3, label='Part of Series', alpha=0.7),
    plt.Line2D([0], [0], color='#8E8E93', linewidth=2, label='Secondary Connections', alpha=0.4, linestyle='dashed'),
]

ax.legend(handles=legend_elements, loc='upper center',
         bbox_to_anchor=(0.5, -0.02),
         frameon=True, fancybox=True, shadow=True,
         ncol=5, fontsize=11)
ax.axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + f"{TARGET_ENTITY}_network_graph.png",
           bbox_inches='tight', dpi=300, facecolor='white')
print(f"Saved: {OUTPUT_FOLDER}{TARGET_ENTITY}_network_graph.png")
plt.close()
# ============================================================================
# 6. TEMPORAL EVOLUTION (SEPARATE)
# ============================================================================
print("\nCreating temporal evolution chart...")

if temporal_data:
    years = sorted([y for y in temporal_data.keys() if y and y.isdigit()])

    if years:
        fig, ax = plt.subplots(figsize=(14, 6))

        years_int = [int(y) for y in years]
        mentions_data = [temporal_data[y].get('mentions', 0) for y in years]
        tags_data = [temporal_data[y].get('hasTag', 0) for y in years]
        series_data = [temporal_data[y].get('partOfSeries', 0) for y in years]

        x = np.arange(len(years))
        width = 0.25

        bars1 = ax.bar(x - width, mentions_data, width, label='Mentions',
                      color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x, tags_data, width, label='Has Tag',
                      color='#4ECDC4', alpha=0.8)
        bars3 = ax.bar(x + width, series_data, width, label='Part of Series',
                      color='#95E77E', alpha=0.8)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Meme Connections', fontsize=12)
        ax.set_title(f'Temporal Evolution: {entity_label} in Meme Culture',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(years_int, rotation=45 if len(years) > 10 else 0)
        ax.legend(loc='upper left', frameon=True, fancybox=True)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(OUTPUT_FOLDER + f"{TARGET_ENTITY}_temporal_evolution.png",
                   bbox_inches='tight', dpi=300)
        print(f"Saved: {OUTPUT_FOLDER}{TARGET_ENTITY}_temporal_evolution.png")
        plt.close()

# ============================================================================
# 7. CONNECTION TYPE ANALYSIS (SEPARATE)
# ============================================================================
print("\nCreating connection type analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart of connection types
sizes = [len(memes) for memes in connection_breakdown.values()]
colors = ['#FF6B6B', '#4ECDC4', '#95E77E']
explode = (0.05, 0.05, 0.05)

wedges, texts, autotexts = ax1.pie(sizes, labels=connection_breakdown.keys(),
                                    colors=colors, autopct='%1.1f%%',
                                    explode=explode, shadow=True, startangle=90)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')

ax1.set_title(f'Distribution of Connection Types\nTotal: {sum(sizes)} connections',
             fontsize=12, fontweight='bold')

# Engagement by connection type
engagement_by_type = {}
for conn_type, memes in connection_breakdown.items():
    total_engagement = sum(m['engagement'] for m in memes)
    avg_engagement = total_engagement / len(memes) if memes else 0
    engagement_by_type[conn_type] = avg_engagement

bars = ax2.bar(engagement_by_type.keys(), engagement_by_type.values(),
              color=colors, alpha=0.8)

for bar, value in zip(bars, engagement_by_type.values()):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{value:,.0f}', ha='center', va='bottom', fontsize=10)

ax2.set_ylabel('Average Engagement Score', fontsize=11)
ax2.set_title('Average Engagement by Connection Type', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + f"{TARGET_ENTITY}_connection_type_analysis.png",
           bbox_inches='tight', dpi=300)
print(f"Saved: {OUTPUT_FOLDER}{TARGET_ENTITY}_connection_type_analysis.png")
plt.close()

# ============================================================================
# 8. CO-OCCURRENCE HEATMAP (SEPARATE)
# ============================================================================
print("\nGenerating co-occurrence heatmap...")

# Build co-occurrence matrix for top entities
top_entities = [data['display_label'] for _, data in top_secondary[:10]]
cooccurrence_matrix = np.zeros((len(top_entities), len(top_entities)))

for i, entity1 in enumerate(top_entities):
    for j, entity2 in enumerate(top_entities):
        if i != j:
            # Find normalized versions
            entity1_norm = None
            entity2_norm = None

            for norm, data in entity_connections.items():
                if data['display_label'] == entity1:
                    entity1_norm = norm
                if data['display_label'] == entity2:
                    entity2_norm = norm

            # Count shared memes
            if entity1_norm and entity2_norm:
                shared_memes = entity_connections[entity1_norm]['memes'].intersection(
                    entity_connections[entity2_norm]['memes']
                )
                cooccurrence_matrix[i, j] = len(shared_memes)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cooccurrence_matrix,
            xticklabels=top_entities,
            yticklabels=top_entities,
            annot=True,
            fmt='g',
            cmap='YlOrRd',
            cbar_kws={'label': 'Shared Meme Connections'},
            ax=ax)

ax.set_title(f'Entity Co-occurrence Matrix: Entities Connected via {entity_label} Memes',
            fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + f"{TARGET_ENTITY}_cooccurrence_heatmap.png",
           bbox_inches='tight', dpi=300)
print(f"Saved: {OUTPUT_FOLDER}{TARGET_ENTITY}_cooccurrence_heatmap.png")
plt.close()

# ============================================================================
# 9. PATHWAY ANALYSIS (SEPARATE)
# ============================================================================
print("\nAnalyzing connection pathways...")

# Analyze the pathways through which entities connect
pathway_stats = defaultdict(lambda: {'entities': set(), 'total': 0})

for _, data in entity_connections.items():
    for pathway, count in data['connection_paths'].items():
        pathway_stats[pathway]['entities'].add(data['display_label'])
        pathway_stats[pathway]['total'] += count

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Pathway distribution
pathways = list(pathway_stats.keys())
pathway_counts = [pathway_stats[p]['total'] for p in pathways]
pathway_diversity = [len(pathway_stats[p]['entities']) for p in pathways]

ax1.barh(pathways, pathway_counts, color='steelblue', alpha=0.8)
ax1.set_xlabel('Total Connections', fontsize=11)
ax1.set_title(f'Connection Pathways from {entity_label} to Other Entities',
             fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

for i, count in enumerate(pathway_counts):
    ax1.text(count + 0.5, i, str(count), va='center', fontsize=9)

ax2.barh(pathways, pathway_diversity, color='coral', alpha=0.8)
ax2.set_xlabel('Unique Entities Reached', fontsize=11)
ax2.set_title(f'Pathway Diversity: Unique Entities per Connection Type',
             fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for i, count in enumerate(pathway_diversity):
    ax2.text(count + 0.2, i, str(count), va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_FOLDER + f"{TARGET_ENTITY}_pathway_analysis.png",
           bbox_inches='tight', dpi=300)
print(f"Saved: {OUTPUT_FOLDER}{TARGET_ENTITY}_pathway_analysis.png")
plt.close()

# ============================================================================
# 10. GENERATE SUMMARY STATISTICS
# ============================================================================
print("\nGenerating summary statistics...")

total_views = sum(m['views'] for m in meme_data)
total_comments = sum(m['comments'] for m in meme_data)
avg_engagement = np.mean([m['engagement'] for m in meme_data if m['engagement'] > 0])

summary = {
    'Target Entity': entity_label,
    'Entity Type': entity_type,
    'Total MemeEntry connections': len(meme_results),
    'Connections via mentions': len(connection_breakdown.get('mentions', [])),
    'Connections via hasTag': len(connection_breakdown.get('hasTag', [])),
    'Connections via partOfSeries': len(connection_breakdown.get('partOfSeries', [])),
    'Unique connected entities (case-insensitive)': len(entity_connections),
    'Total Views': f'{total_views:,}',
    'Total Comments': f'{total_comments:,}',
    'Average Engagement': f'{avg_engagement:,.0f}',
    'Network Nodes': G.number_of_nodes(),
    'Network Edges': G.number_of_edges(),
}

# Save summary
with open(OUTPUT_FOLDER + f"{TARGET_ENTITY}_summary_statistics.txt", "w") as f:
    f.write(f"SEMANTIC QUERY SUMMARY: {entity_label.upper()} IN MEME CULTURE\n")
    f.write("=" * 60 + "\n\n")
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")

    f.write("\n\nTOP CONNECTED ENTITIES (CASE-INSENSITIVE):\n")
    f.write("-" * 40 + "\n")
    for i, (_, data) in enumerate(top_secondary[:15], 1):
        f.write(f"{i}. {data['display_label']} ({list(data['types'])[0]}) - {data['count']} connections\n")

print(f"Saved: {OUTPUT_FOLDER}{TARGET_ENTITY}_summary_statistics.txt")

print("\n" + "=" * 80)
print("COMPLETE SEMANTIC QUERY ANALYSIS FINISHED")
print("=" * 80)
print(f"\nGenerated visualizations in: {OUTPUT_FOLDER}")
print("\nOutputs:")
print(f"1. {TARGET_ENTITY}_entity_entry_details.png - Entity RDF details panel")
print(f"2. {TARGET_ENTITY}_network_graph.png - Full network with secondary connections")
print(f"3. {TARGET_ENTITY}_temporal_evolution.png - Temporal bar chart")
print(f"4. {TARGET_ENTITY}_connection_type_analysis.png - Pie chart and engagement")
print(f"5. {TARGET_ENTITY}_cooccurrence_heatmap.png - Entity co-occurrence matrix")
print(f"6. {TARGET_ENTITY}_pathway_analysis.png - Connection pathways")
print(f"7. {TARGET_ENTITY}_summary_statistics.txt - Complete statistics")