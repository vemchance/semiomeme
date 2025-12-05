from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD
import networkx as nx
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import re
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning,
                        message='.*Glyph.*missing from current font.*')

from kgraph.analysis.output_manager import AnalysisOutputManager
from config.namespaces import SMO, EX, SCHEMA


class OntologyAnalyser:
    """A tool for analysing hierarchical RDF graphs with proper ontology structure."""

    def __init__(self, ttl_path: str, output_base_dir: str = "outputs/analysis", use_timestamp: bool = True):
        """Initialise with path to turtle file."""
        self.graph_path = ttl_path
        self.g = Graph()
        self.g.parse(ttl_path, format='turtle')
        print(f"Loaded graph with {len(self.g)} triples")

        if use_timestamp:
            # Use output manager with timestamp
            self.output_manager = AnalysisOutputManager(output_base_dir)
            self.output_dir = self.output_manager.create_analysis_session("ontology")
        else:
            # Use directory directly without timestamp
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            self.output_dir = project_root / output_base_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.output_manager = None

        # Initialize namespaces from config
        self.SMO = SMO
        self.EX = EX
        self.SCHEMA = SCHEMA

        self.g.bind('smo', self.SMO)
        self.g.bind('ex', self.EX)
        self.g.bind('schema', self.SCHEMA)
        self.g.bind('owl', OWL)


        # Analysis results
        self.entity_counts = {}
        self.property_counts = {}
        self.property_domains = {}
        self.property_ranges = {}
        self.class_relationships = {}
        self.instance_classes = {}
        self.anomalies = []

    def analyse_graph(self):
        """Run complete analysis on the semiomeme ontology graph."""
        print("Beginning semiomeme graph analysis...")

        self.count_entity_types()
        self.analyse_property_usage()
        self.detect_class_relationships()
        self.find_property_domains_and_ranges()
        self.detect_anomalies()

        print("Analysis complete!")

    def count_entity_types(self):
        """Count instances of each SMO ontology class."""
        print("Counting entity types...")

        # Count instances by their SMO type
        type_counts = defaultdict(int)

        for s, p, o in self.g.triples((None, RDF.type, None)):
            # Only count SMO ontology types (the actual classes)
            if isinstance(o, URIRef) and str(o).startswith(str(self.SMO)):
                type_counts[o] += 1

        self.entity_counts = dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True))

        print(f"Found {len(self.entity_counts)} SMO ontology classes with instances")
        print(f"Total instances: {sum(self.entity_counts.values())}")

        # Show top classes
        for class_uri, count in list(self.entity_counts.items())[:10]:
            class_name = str(class_uri).replace(str(self.SMO), 'smo:')
            print(f"  {class_name}: {count}")

        # Save results
        with open(f"{self.output_dir}/entity_types.csv", 'w', encoding='utf-8') as f:
            f.write("Type,Count,Category\n")
            for type_uri, count in self.entity_counts.items():
                f.write(f"{type_uri},{count},ontology_class\n")

        # Create visualisation
        if self.entity_counts:
            plt.figure(figsize=(12, 8))

            # Get top 15 classes
            top_types = list(self.entity_counts.items())[:15]
            labels = [str(t).replace(str(self.SMO), 'smo:') for t, c in top_types]
            values = [c for t, c in top_types]

            plt.barh(labels, values, color='lightblue', edgecolor='darkblue')
            plt.title('SMO Ontology Classes by Instance Count')
            plt.xlabel('Number of Instances')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/entity_types.png", dpi=300, bbox_inches='tight')

    def analyse_property_usage(self):
        """Analyse property usage across the graph."""
        print("Analysing property usage...")

        # Skip only the most basic RDF/RDFS properties
        skip_properties = {
            RDF.type,  # Type assertions are counted separately
            RDFS.subClassOf,  # Ontology structure
            RDFS.domain,  # Ontology definitions
            RDFS.range,  # Ontology definitions
            RDFS.comment  # Comments
        }

        property_counts = defaultdict(int)
        smo_properties = set()

        for s, p, o in self.g:
            if p not in skip_properties:
                property_counts[p] += 1

                # Track SMO properties
                if str(p).startswith(str(self.SMO)):
                    smo_properties.add(p)

        self.property_counts = dict(sorted(property_counts.items(), key=lambda x: x[1], reverse=True))

        print(f"Found {len(self.property_counts)} total properties")
        print(f"Found {len(smo_properties)} SMO ontology properties")

        # Show top properties
        for prop, count in list(self.property_counts.items())[:10]:
            prop_name = self._format_property_name(prop)
            print(f"  {prop_name}: {count}")

        # Save results
        with open(f"{self.output_dir}/property_usage.csv", 'w', encoding='utf-8') as f:
            f.write("Property,Count,IsOntologyProperty\n")
            for prop, count in self.property_counts.items():
                is_ontology = str(prop).startswith(str(self.SMO))
                f.write(f"{prop},{count},{is_ontology}\n")

        # Create visualisation
        if self.property_counts:
            plt.figure(figsize=(14, 10))
            top_props = list(self.property_counts.items())[:20]
            labels = [self._format_property_name(p) for p, c in top_props]
            values = [c for p, c in top_props]

            # Colour code: SMO properties vs others
            colors = ['lightgreen' if str(p).startswith(str(self.SMO)) else 'lightcoral'
                      for p, c in top_props]

            plt.barh(labels, values, color=colors, edgecolor='black', linewidth=0.5)
            plt.title('Top 20 Properties (Green = SMO Ontology Properties)')
            plt.xlabel('Usage Count')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/property_usage.png", dpi=300, bbox_inches='tight')

    def find_property_domains_and_ranges(self):
        """Analyse property domains and ranges."""
        print("Analysing property domains and ranges...")

        # Extract explicit domain/range definitions
        self.property_domains = {}
        self.property_ranges = {}

        for prop, p, domain in self.g.triples((None, RDFS.domain, None)):
            self.property_domains[prop] = domain
            print(f"Explicit domain: {self._format_property_name(prop)} → {self._format_class_name(domain)}")

        for prop, p, range_val in self.g.triples((None, RDFS.range, None)):
            self.property_ranges[prop] = range_val
            print(f"Explicit range: {self._format_property_name(prop)} → {self._format_class_name(range_val)}")

        # Infer domains and ranges from actual usage
        usage_domains = defaultdict(lambda: defaultdict(int))
        usage_ranges = defaultdict(lambda: defaultdict(int))

        for s, p, o in self.g:
            if p == RDF.type or p in self.property_domains:
                continue

            # Get subject types for domain inference
            for _, _, s_type in self.g.triples((s, RDF.type, None)):
                if str(s_type).startswith(str(self.SMO)):  # Only SMO types
                    usage_domains[p][s_type] += 1

            # Get object types for range inference
            if p not in self.property_ranges:
                if isinstance(o, URIRef):
                    for _, _, o_type in self.g.triples((o, RDF.type, None)):
                        if str(o_type).startswith(str(self.SMO)):  # Only SMO types
                            usage_ranges[p][o_type] += 1
                elif isinstance(o, Literal):
                    datatype = o.datatype or XSD.string
                    usage_ranges[p][datatype] += 1

        # Infer most common domains/ranges for properties without explicit definitions
        for prop in usage_domains:
            if prop not in self.property_domains and usage_domains[prop]:
                # Only infer if used at least 3 times
                total_usage = sum(usage_domains[prop].values())
                if total_usage >= 3:
                    most_common = max(usage_domains[prop].items(), key=lambda x: x[1])
                    # Only infer if this domain represents at least 70% of usage
                    if most_common[1] / total_usage >= 0.7:
                        self.property_domains[prop] = most_common[0]

        for prop in usage_ranges:
            if prop not in self.property_ranges and usage_ranges[prop]:
                total_usage = sum(usage_ranges[prop].values())
                if total_usage >= 3:
                    most_common = max(usage_ranges[prop].items(), key=lambda x: x[1])
                    if most_common[1] / total_usage >= 0.7:
                        self.property_ranges[prop] = most_common[0]

        # Save results with detailed analysis
        with open(f"{self.output_dir}/property_domains_ranges.csv", 'w', encoding='utf-8') as f:
            f.write(
                "Property,Domain,Range,DomainSource,RangeSource,UsageCount,DomainConfidence,RangeConfidence,TopDomains,TopRanges\n")
            for prop in self.property_counts:
                domain = self.property_domains.get(prop, "None")
                range_val = self.property_ranges.get(prop, "None")
                domain_source = "explicit" if prop in [p for p, _, _ in
                                                       self.g.triples((None, RDFS.domain, None))] else "inferred"
                range_source = "explicit" if prop in [p for p, _, _ in
                                                      self.g.triples((None, RDFS.range, None))] else "inferred"

                usage_count = self.property_counts[prop]

                # Calculate confidence and alternatives for properties without explicit definitions
                domain_confidence = ""
                range_confidence = ""
                top_domains = ""
                top_ranges = ""

                if prop not in [p for p, _, _ in self.g.triples((None, RDFS.domain, None))]:
                    # Analyse domain usage
                    domain_usage = defaultdict(int)
                    for s, p_used, o in self.g.triples((None, prop, None)):
                        for _, _, s_type in self.g.triples((s, RDF.type, None)):
                            if str(s_type).startswith(str(self.SMO)):
                                domain_usage[s_type] += 1

                    if domain_usage:
                        total_domain_usage = sum(domain_usage.values())
                        sorted_domains = sorted(domain_usage.items(), key=lambda x: x[1], reverse=True)

                        if domain in domain_usage:
                            domain_confidence = f"{domain_usage[domain] / total_domain_usage:.1%}"

                        # Show top 3 domains with percentages
                        top_domain_list = []
                        for d, count in sorted_domains[:3]:
                            d_name = self._format_class_name(d)
                            percentage = count / total_domain_usage
                            top_domain_list.append(f"{d_name}({percentage:.0%})")
                        top_domains = "; ".join(top_domain_list)

                if prop not in [p for p, _, _ in self.g.triples((None, RDFS.range, None))]:
                    # Analyse range usage
                    range_usage = defaultdict(int)
                    for s, p_used, o in self.g.triples((None, prop, None)):
                        if isinstance(o, URIRef):
                            for _, _, o_type in self.g.triples((o, RDF.type, None)):
                                if str(o_type).startswith(str(self.SMO)):
                                    range_usage[o_type] += 1
                        elif isinstance(o, Literal):
                            datatype = o.datatype or XSD.string
                            range_usage[datatype] += 1

                    if range_usage:
                        total_range_usage = sum(range_usage.values())
                        sorted_ranges = sorted(range_usage.items(), key=lambda x: x[1], reverse=True)

                        if range_val in range_usage:
                            range_confidence = f"{range_usage[range_val] / total_range_usage:.1%}"

                        # Show top 3 ranges with percentages
                        top_range_list = []
                        for r, count in sorted_ranges[:3]:
                            r_name = self._format_class_name(r)
                            percentage = count / total_range_usage
                            top_range_list.append(f"{r_name}({percentage:.0%})")
                        top_ranges = "; ".join(top_range_list)

                f.write(
                    f'"{prop}","{domain}","{range_val}",{domain_source},{range_source},{usage_count},"{domain_confidence}","{range_confidence}","{top_domains}","{top_ranges}"\n')

    def detect_class_relationships(self):
        """Detect class relationships in the SMO ontology."""
        print("Detecting class relationships...")

        # Get explicit subclass relationships
        explicit_subclasses = {}
        for subclass, p, superclass in self.g.triples((None, RDFS.subClassOf, None)):
            if subclass not in explicit_subclasses:
                explicit_subclasses[subclass] = set()
            explicit_subclasses[subclass].add(superclass)
            print(f"Explicit hierarchy: {self._format_class_name(subclass)} ⊆ {self._format_class_name(superclass)}")

        # Get instances by SMO type for overlap analysis
        instances_by_type = defaultdict(set)
        for s, _, o in self.g.triples((None, RDF.type, None)):
            if str(o).startswith(str(self.SMO)):
                instances_by_type[o].add(s)

        # Find potential relationships based on instance overlap
        inferred_relationships = defaultdict(set)
        for type1 in instances_by_type:
            for type2 in instances_by_type:
                if type1 != type2:
                    instances1 = instances_by_type[type1]
                    instances2 = instances_by_type[type2]

                    overlap = instances1.intersection(instances2)
                    if overlap and len(instances1) > 2:  # At least 3 instances
                        confidence = len(overlap) / len(instances1)
                        if confidence > 0.5:  # Significant overlap
                            inferred_relationships[type1].add((type2, confidence))

        # Combine results
        self.class_relationships = {}

        # Add explicit relationships
        for subclass, superclasses in explicit_subclasses.items():
            self.class_relationships[subclass] = {sc: 1.0 for sc in superclasses}

        # Add high-confidence inferred relationships
        for cls, relations in inferred_relationships.items():
            if cls not in self.class_relationships:
                self.class_relationships[cls] = {}
            for parent, conf in relations:
                if conf > 0.8:  # Very high confidence only
                    self.class_relationships[cls][parent] = conf

        # Save results
        with open(f"{self.output_dir}/class_relationships.csv", 'w', encoding='utf-8') as f:
            f.write("Class,Parent,Confidence,Source\n")
            for cls, relations in self.class_relationships.items():
                for parent, conf in relations.items():
                    source = "explicit" if conf == 1.0 else "inferred"
                    f.write(f"{cls},{parent},{conf:.2f},{source}\n")

    def detect_anomalies(self):
        """Detect structural anomalies in the semiomeme graph with improved logic."""
        print("Detecting anomalies...")

        # 1. Entities with multiple SMO types - now filtering for problematic cases only
        problematic_multi_typed_entities = 0

        # Define acceptable multi-type combinations
        acceptable_combinations = {
            ('Tag', 'MemeEntry'),
            ('Tag', 'PersonEntry'),
            ('Tag', 'EventEntry'),
            ('Tag', 'SiteEntry'),
            ('Tag', 'SubcultureEntry'),
            ('Tag', 'LocationEntry'),
            ('Tag', 'OrganizationEntry'),
            ('Tag', 'Entity'),
            ('Tag', 'Series'),
            # Series can contain other types
            ('Series', 'MemeEntry'),
            ('Series', 'EventEntry'),
            # Entity is a generic fallback
            ('Entity', 'MemeEntry'),
            ('Entity', 'PersonEntry'),
            ('Entity', 'EventEntry'),
        }

        for s in set(s for s, p, o in self.g.triples((None, RDF.type, None))):
            smo_types = [o for _, _, o in self.g.triples((s, RDF.type, None))
                         if str(o).startswith(str(self.SMO))]

            if len(smo_types) > 1:
                # Extract class names for comparison
                type_names = set()
                for t in smo_types:
                    class_name = str(t).split('/')[-1]
                    # Remove 'Entry' suffix for comparison
                    simplified_name = class_name.replace('Entry', '') if class_name.endswith('Entry') else class_name
                    type_names.add(simplified_name)

                # Check if this combination is acceptable
                is_acceptable = False
                for combo in acceptable_combinations:
                    if type_names == set(combo):
                        is_acceptable = True
                        break

                if not is_acceptable:
                    problematic_multi_typed_entities += 1
                    type_labels = [self._format_class_name(t) for t in smo_types]
                    self.anomalies.append({
                        "entity": str(s),
                        "issue": "problematic_multiple_smo_types",
                        "details": f"Entity has unexpected type combination: {', '.join(type_labels)}"
                    })

        # 2. Severe domain violations only - ignore minor inheritance issues
        severe_domain_violations = 0
        for prop, domain in self.property_domains.items():
            if str(prop).startswith(str(self.SMO)):
                violations = 0
                total_usage = 0

                for s, p, o in self.g.triples((None, prop, None)):
                    total_usage += 1
                    s_types = list(self.g.objects(s, RDF.type))
                    smo_types = [t for t in s_types if str(t).startswith(str(self.SMO))]

                    # Skip if entity has no SMO types (these are handled elsewhere)
                    if not smo_types:
                        continue

                    # Check direct match
                    if domain in s_types:
                        continue

                    # Check subclass relationships
                    is_valid = False
                    for s_type in smo_types:
                        if (s_type, RDFS.subClassOf, domain) in self.g:
                            is_valid = True
                            break

                    # Allow Tag entities to use most properties (as they're multi-purpose)
                    tag_type = URIRef(str(self.SMO) + 'Tag')
                    if tag_type in smo_types:
                        is_valid = True

                    if not is_valid:
                        violations += 1

                # Only flag as severe if >50% of usage violates domain and it's used frequently
                if violations > 0 and total_usage > 10:
                    violation_rate = violations / total_usage
                    if violation_rate > 0.5:
                        severe_domain_violations += violations
                        self.anomalies.append({
                            "property": str(prop),
                            "issue": "severe_domain_violation",
                            "details": f"Property has {violations}/{total_usage} severe domain violations ({violation_rate:.1%}) - domain: {self._format_class_name(domain)}"
                        })

        # 3. EX entities without SMO types - unchanged as this is still problematic
        untyped_ex_entities = 0
        for s, p, o in self.g:
            if isinstance(s, URIRef) and str(s).startswith(str(self.EX)):
                smo_types = [t for t in self.g.objects(s, RDF.type) if str(t).startswith(str(self.SMO))]
                if not smo_types:
                    untyped_ex_entities += 1

        if untyped_ex_entities > 0:
            self.anomalies.append({
                "issue": "untyped_ex_entities",
                "details": f"Found {untyped_ex_entities} EX entities without SMO types"
            })

        # 4. Critical singleton properties - only flag truly unused properties
        critical_singleton_props = []
        for prop, count in self.property_counts.items():
            if str(prop).startswith(str(self.SMO)) and count == 1:
                # Skip automatically generated properties from REBEL extraction
                prop_name = str(prop).split('/')[-1]
                if not any(pattern in prop_name.lower() for pattern in
                           ['mentions', 'instance_of', 'located_in', 'part_of']):
                    critical_singleton_props.append(prop)

        if critical_singleton_props:
            self.anomalies.append({
                "issue": "critical_singleton_properties",
                "details": f"Found {len(critical_singleton_props)} manually-defined SMO properties used only once"
            })

        print(f"Found {len(self.anomalies)} structural issues:")
        print(f"  - {problematic_multi_typed_entities} entities with problematic type combinations")
        print(f"  - {severe_domain_violations} severe domain violations")
        print(f"  - {untyped_ex_entities} untyped EX entities")
        print(f"  - {len(critical_singleton_props)} critical singleton properties")

        # Save anomalies
        with open(f"{self.output_dir}/anomalies.json", 'w', encoding='utf-8') as f:
            json.dump(self.anomalies, f, indent=2)

    def visualise_graph_structure(self):
        """Create visualisation of SMO ontology structure with properly sized nodes."""
        print("Generating ontology structure visualisation...")

        G = nx.DiGraph()

        # Add SMO class hierarchy
        for subclass, relations in self.class_relationships.items():
            for superclass, conf in relations.items():
                if conf > 0.7:  # Only strong relationships
                    sub_label = self._format_class_name(subclass)
                    super_label = self._format_class_name(superclass)
                    G.add_edge(super_label, sub_label, weight=conf, type='subclass')

        # Add property domain/range relationships for major SMO properties only
        for prop, count in self.property_counts.items():
            if count > 50 and str(prop).startswith(str(self.SMO)):  # Increased threshold
                prop_label = self._format_property_name(prop)

                if prop in self.property_domains:
                    domain_label = self._format_class_name(self.property_domains[prop])
                    G.add_edge(domain_label, prop_label, type='domain')

                if prop in self.property_ranges:
                    range_uri = self.property_ranges[prop]
                    if not str(range_uri).startswith('http://www.w3.org/2001/XMLSchema'):
                        range_label = self._format_class_name(range_uri)
                        G.add_edge(prop_label, range_label, type='range')

        if not G.nodes():
            print("Warning: No relationships found for visualisation")
            return

        # Set node attributes
        for node in G.nodes():
            # Determine if it's a class or property
            if any(node == self._format_class_name(cls) for cls in self.entity_counts.keys()):
                G.nodes[node]['type'] = 'class'
                # Find the count
                for cls, count in self.entity_counts.items():
                    if node == self._format_class_name(cls):
                        G.nodes[node]['count'] = count
                        break
            elif any(node == self._format_property_name(prop) for prop in self.property_counts.keys()):
                G.nodes[node]['type'] = 'property'
                # Find the count
                for prop, count in self.property_counts.items():
                    if node == self._format_property_name(prop):
                        G.nodes[node]['count'] = count
                        break
            else:
                G.nodes[node]['type'] = 'external'
                G.nodes[node]['count'] = 1

        # Create visualisation with better layout
        plt.figure(figsize=(24, 18))  # Larger figure

        # Use hierarchical layout for better structure visibility
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            # Fallback to spring layout with better parameters
            pos = nx.spring_layout(G, k=8, iterations=100, seed=42)

        # Separate nodes by type
        class_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'class']
        prop_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'property']
        external_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'external']

        # FIXED: Much smaller, logarithmic node sizing
        import math

        # Calculate size ranges
        if class_nodes:
            max_class_count = max(G.nodes[n].get('count', 1) for n in class_nodes)
            min_class_count = min(G.nodes[n].get('count', 1) for n in class_nodes)

            class_sizes = []
            for n in class_nodes:
                count = G.nodes[n].get('count', 1)
                # Logarithmic scaling: 300-1500 range
                if max_class_count > min_class_count:
                    norm_count = (math.log(count + 1) - math.log(min_class_count + 1)) / \
                                 (math.log(max_class_count + 1) - math.log(min_class_count + 1))
                    size = 300 + norm_count * 1200
                else:
                    size = 800
                class_sizes.append(size)
        else:
            class_sizes = []

        if prop_nodes:
            max_prop_count = max(G.nodes[n].get('count', 1) for n in prop_nodes)
            min_prop_count = min(G.nodes[n].get('count', 1) for n in prop_nodes)

            prop_sizes = []
            for n in prop_nodes:
                count = G.nodes[n].get('count', 1)
                # Smaller range for properties: 200-800
                if max_prop_count > min_prop_count:
                    norm_count = (math.log(count + 1) - math.log(min_prop_count + 1)) / \
                                 (math.log(max_prop_count + 1) - math.log(min_prop_count + 1))
                    size = 200 + norm_count * 600
                else:
                    size = 400
                prop_sizes.append(size)
        else:
            prop_sizes = []

        external_sizes = [250 for n in external_nodes]  # Fixed small size

        # Draw nodes
        if class_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=class_nodes, node_color='lightblue',
                                   node_size=class_sizes, alpha=0.8, edgecolors='darkblue', linewidths=1.5)

        if prop_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=prop_nodes, node_color='lightgreen',
                                   node_size=prop_sizes, alpha=0.8, edgecolors='darkgreen', linewidths=1.5)

        if external_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=external_nodes, node_color='lightgrey',
                                   node_size=external_sizes, alpha=0.7, edgecolors='grey', linewidths=1)

        # Draw edges by type with reduced width
        edge_types = {}
        for u, v, d in G.edges(data=True):
            edge_type = d.get('type', 'unknown')
            if edge_type not in edge_types:
                edge_types[edge_type] = []
            edge_types[edge_type].append((u, v))

        edge_styles = {
            'subclass': {'color': 'blue', 'style': '-', 'width': 2, 'label': 'subClassOf'},
            'domain': {'color': 'green', 'style': '--', 'width': 1.5, 'label': 'domain'},
            'range': {'color': 'red', 'style': ':', 'width': 1.5, 'label': 'range'}
        }

        for edge_type, edges in edge_types.items():
            if edge_type in edge_styles:
                style = edge_styles[edge_type]
                nx.draw_networkx_edges(G, pos, edgelist=edges,
                                       edge_color=style['color'],
                                       style=style['style'],
                                       width=style['width'],
                                       arrows=True, arrowsize=15, alpha=0.7)

        # Add labels with better font size
        label_dict = {}
        for node in G.nodes():
            # Truncate long labels
            if len(node) > 15:
                label_dict[node] = node[:12] + "..."
            else:
                label_dict[node] = node

        nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=7, font_weight='bold')

        # Create legend with better positioning
        legend_elements = []
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                                          markersize=10, markeredgecolor='darkblue', label='SMO Class'))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                                          markersize=8, markeredgecolor='darkgreen', label='SMO Property'))

        for edge_type, style in edge_styles.items():
            if edge_type in edge_types:
                legend_elements.append(plt.Line2D([0], [0], color=style['color'],
                                                  linewidth=style['width'], linestyle=style['style'],
                                                  label=f"{style['label']} ({len(edge_types[edge_type])})"))

        plt.legend(handles=legend_elements, loc='upper left', title='SMO Ontology Elements', fontsize=10)
        plt.axis('off')
        plt.title('SemioMeme Ontology Structure', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/ontology_structure.png", dpi=300, bbox_inches='tight')

        print(f"Ontology structure saved with {len(G.nodes())} nodes and {len(G.edges())} edges")

    def visualise_entity_subgraph(self, entity_label=None, entity_uri=None, random_entity=False,
                                  entity_type_filter=None, max_depth=2, max_nodes=50,
                                  direction='both', exclude_types=None, include_types=None,
                                  exclude_properties=None, include_properties=None,
                                  exclude_literals=True, min_connections=0):
        """
        Create a detailed visualisation of a single entity and its immediate connections.

        Args:
            entity_label: Human-readable label to search for (e.g., "Donald Trump")
            entity_uri: Specific entity URI to visualise
            random_entity: If True, select a random entity to visualise
            entity_type_filter: When using random_entity, filter by SMO type (e.g., "PersonEntry", "MemeEntry")
            max_depth: Maximum relationship depth to traverse (1=direct, 2=2-hop)
            max_nodes: Maximum nodes to include (prevents overcrowding)
            direction: 'outgoing', 'incoming', or 'both' - which relationships to follow
            exclude_types: List of SMO types to exclude (e.g., ['Tag', 'Status'])
            include_types: List of SMO types to include only (overrides exclude_types)
            exclude_properties: List of properties to ignore (e.g., ['hasTag', 'hasStatus'])
            include_properties: List of properties to include only
            exclude_literals: If True, don't show literal values as nodes
            min_connections: Only show connected entities with at least N total connections
        """
        print(f"Generating entity subgraph visualisation...")

        # Find the target entity
        target_entity = None
        target_label_str = None
        target_type = None

        if random_entity:
            import random
            candidates = []
            type_filter = URIRef(self.SMO + entity_type_filter) if entity_type_filter else None

            for s, p, o in self.g.triples((None, RDF.type, type_filter)):
                if str(s).startswith(str(self.EX)):
                    labels = list(self.g.objects(s, RDFS.label))
                    label = str(labels[0]) if labels else str(s).split('/')[-1]

                    smo_types = [t for t in self.g.objects(s, RDF.type) if str(t).startswith(str(self.SMO))]
                    if smo_types:
                        candidates.append((s, label, smo_types[0]))

            if candidates:
                target_entity, target_label_str, target_type = random.choice(candidates)
                print(f"Randomly selected: {target_label_str} ({self._format_class_name(target_type)})")
            else:
                print(f"No entities found" + (f" of type {entity_type_filter}" if entity_type_filter else ""))
                return

        elif entity_uri:
            target_entity = URIRef(entity_uri)

            labels = list(self.g.objects(target_entity, RDFS.label))
            target_label_str = str(labels[0]) if labels else str(target_entity).split('/')[-1]

            smo_types = [t for t in self.g.objects(target_entity, RDF.type) if str(t).startswith(str(self.SMO))]
            if not smo_types:
                print(f"Entity {target_label_str} has no SMO type")
                return
            target_type = smo_types[0]

        elif entity_label:
            for s, p, o in self.g.triples((None, RDFS.label, None)):
                if isinstance(o, Literal) and entity_label.lower() in str(o).lower():
                    target_entity = s
                    target_label_str = str(o)

                    smo_types = [t for t in self.g.objects(s, RDF.type) if str(t).startswith(str(self.SMO))]
                    if smo_types:
                        target_type = smo_types[0]
                        print(f"Found entity: {target_label_str} ({self._format_class_name(target_type)})")
                        break
                    else:
                        continue

        if not target_entity or not target_type:
            if entity_label:
                print(f"Entity with label containing '{entity_label}' and SMO type not found")
            else:
                print("No valid entity specified")
            return

        print(f"Processing target entity: {target_label_str} (type: {self._format_class_name(target_type)})")

        # Apply filters
        if exclude_types is None:
            exclude_types = []
        if exclude_properties is None:
            exclude_properties = []

        excluded_type_uris = [URIRef(self.SMO + t) if not str(t).startswith('http') else URIRef(t)
                              for t in exclude_types]
        excluded_prop_uris = [URIRef(self.SMO + p) if not str(p).startswith('http') else URIRef(p)
                              for p in exclude_properties]

        # Build subgraph using NetworkX
        G = nx.Graph()

        # Add target entity first with validated information
        G.add_node(target_label_str,
                   uri=str(target_entity),
                   type=self._format_class_name(target_type),
                   depth=0,
                   is_target=True)

        # Track entities to process and their depths
        entities_to_process = [(target_entity, target_label_str, target_type, 0)]
        processed_entities = set([target_entity])

        while entities_to_process and len(G.nodes()) < max_nodes:
            current_entity, current_label, current_type, depth = entities_to_process.pop(0)

            if depth > max_depth:
                continue

            # Process outgoing relationships if requested
            if direction in ['outgoing', 'both']:
                for s, p, o in self.g.triples((current_entity, None, None)):
                    if p in [RDF.type, RDFS.label]:
                        continue

                    # Apply property filters
                    if include_properties and str(p) not in include_properties:
                        continue
                    if p in excluded_prop_uris:
                        continue

                    prop_label = self._format_property_name(p)

                    if isinstance(o, URIRef) and str(o).startswith(str(self.EX)):
                        if o in processed_entities:
                            continue

                        o_labels = list(self.g.objects(o, RDFS.label))
                        o_label_str = str(o_labels[0]) if o_labels else str(o).split('/')[-1]

                        o_types = list(self.g.objects(o, RDF.type))
                        o_smo_types = [t for t in o_types if str(t).startswith(str(self.SMO))]

                        if not o_smo_types:
                            continue

                        o_type = o_smo_types[0]

                        # Apply type filters
                        if include_types and self._format_class_name(o_type).replace('smo:', '') not in include_types:
                            continue
                        if o_type in excluded_type_uris:
                            continue

                        # Apply minimum connections filter
                        if min_connections > 0:
                            entity_connection_count = sum(1 for _ in self.g.triples((o, None, None))) + \
                                                      sum(1 for _ in self.g.triples((None, None, o)))
                            if entity_connection_count < min_connections:
                                continue

                        G.add_node(o_label_str,
                                   uri=str(o),
                                   type=self._format_class_name(o_type),
                                   depth=depth + 1,
                                   is_target=False)

                        G.add_edge(current_label, o_label_str,
                                   relation=prop_label,
                                   property_uri=str(p),
                                   direction='outgoing')

                        if depth + 1 <= max_depth:
                            entities_to_process.append((o, o_label_str, o_type, depth + 1))
                        processed_entities.add(o)

                    elif isinstance(o, Literal) and not exclude_literals:
                        # Add important literals as nodes
                        if p in [URIRef(self.SMO + 'yearCreated'), URIRef(self.SMO + 'originDescription')]:
                            literal_label = f"{prop_label}: {str(o)[:30]}..."
                            if literal_label not in G.nodes():
                                G.add_node(literal_label,
                                           uri="literal",
                                           type="Literal",
                                           depth=depth + 1,
                                           is_target=False)
                                G.add_edge(current_label, literal_label,
                                           relation=prop_label,
                                           property_uri=str(p),
                                           direction='outgoing')

            # Process incoming relationships if requested
            if direction in ['incoming', 'both']:
                for s, p, o in self.g.triples((None, None, current_entity)):
                    if p in [RDF.type, RDFS.label] or len(G.nodes()) >= max_nodes:
                        continue

                    # Apply property filters
                    if include_properties and str(p) not in include_properties:
                        continue
                    if p in excluded_prop_uris:
                        continue

                    if isinstance(s, URIRef) and str(s).startswith(str(self.EX)) and s not in processed_entities:
                        prop_label = self._format_property_name(p)

                        s_labels = list(self.g.objects(s, RDFS.label))
                        s_label_str = str(s_labels[0]) if s_labels else str(s).split('/')[-1]

                        s_types = list(self.g.objects(s, RDF.type))
                        s_smo_types = [t for t in s_types if str(t).startswith(str(self.SMO))]

                        if not s_smo_types:
                            continue

                        s_type = s_smo_types[0]

                        # Apply type filters
                        if include_types and self._format_class_name(s_type).replace('smo:', '') not in include_types:
                            continue
                        if s_type in excluded_type_uris:
                            continue

                        # Apply minimum connections filter
                        if min_connections > 0:
                            entity_connection_count = sum(1 for _ in self.g.triples((s, None, None))) + \
                                                      sum(1 for _ in self.g.triples((None, None, s)))
                            if entity_connection_count < min_connections:
                                continue

                        G.add_node(s_label_str,
                                   uri=str(s),
                                   type=self._format_class_name(s_type),
                                   depth=depth + 1,
                                   is_target=False)

                        G.add_edge(s_label_str, current_label,
                                   relation=prop_label,
                                   property_uri=str(p),
                                   direction='incoming')

                        if depth + 1 <= max_depth:
                            entities_to_process.append((s, s_label_str, s_type, depth + 1))
                        processed_entities.add(s)

        if len(G.nodes()) <= 1:
            print("No connections found for this entity")
            return

        # Create visualisation
        plt.figure(figsize=(20, 16))
        pos = nx.spring_layout(G, k=5, iterations=100)

        # Define type colors
        type_colors = {
            'smo:PersonEntry': 'lightcoral',
            'smo:MemeEntry': 'lightblue',
            'smo:EventEntry': 'lightgreen',
            'smo:SiteEntry': 'lightyellow',
            'smo:SubcultureEntry': 'lightsteelblue',
            'smo:LocationEntry': 'lightseagreen',
            'smo:OrganizationEntry': 'lightsalmon',
            'smo:Tag': 'lightpink',
            'smo:Series': 'lightcyan',
            'smo:Status': 'lightgrey',
            'smo:Badge': 'orange',
            'smo:MemeType': 'mediumpurple',
            'smo:Entity': 'wheat',
            'Literal': 'lightgray',
            'smo:MemeFormat': 'lavender',
            'Unknown': 'white'
        }

        # Draw nodes by type
        for node_type, color in type_colors.items():
            nodes_of_type = [n for n, attr in G.nodes(data=True)
                             if attr.get('type', 'Unknown') == node_type and not attr.get('is_target', False)]
            if nodes_of_type:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type,
                                       node_color=color, node_size=800,
                                       alpha=0.8, edgecolors='black', linewidths=1)

        # Draw target entity larger
        target_nodes = [n for n, attr in G.nodes(data=True) if attr.get('is_target', False)]
        if target_nodes:
            target_node_type = G.nodes[target_nodes[0]]['type']
            target_color = type_colors.get(target_node_type, 'white')
            nx.draw_networkx_nodes(G, pos, nodelist=target_nodes,
                                   node_color=target_color, node_size=1500,
                                   alpha=1.0, edgecolors='red', linewidths=4)

        # Draw edges with different styles for incoming/outgoing
        incoming_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr.get('direction') == 'incoming']
        outgoing_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr.get('direction') == 'outgoing']
        other_edges = [(u, v) for u, v, attr in G.edges(data=True) if 'direction' not in attr]

        if incoming_edges:
            nx.draw_networkx_edges(G, pos, edgelist=incoming_edges, alpha=0.6, width=2,
                                   edge_color='blue', style='--', arrows=True, arrowsize=20)
        if outgoing_edges:
            nx.draw_networkx_edges(G, pos, edgelist=outgoing_edges, alpha=0.6, width=2,
                                   edge_color='green', style='-', arrows=True, arrowsize=20)
        if other_edges:
            nx.draw_networkx_edges(G, pos, edgelist=other_edges, alpha=0.6, width=1,
                                   edge_color='gray', arrows=True, arrowsize=15)

        # Add node labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

        # Add edge labels for relationships
        edge_labels = {}
        for u, v, attr in G.edges(data=True):
            if 'relation' in attr:
                edge_labels[(u, v)] = attr['relation']

        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, alpha=0.7)

        # Create legend
        legend_elements = []
        for node_type, color in type_colors.items():
            if any(attr.get('type') == node_type for n, attr in G.nodes(data=True)):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                  markerfacecolor=color, markersize=10,
                                                  markeredgecolor='black', label=node_type))

        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor='white', markersize=15,
                                          markeredgecolor='red', linewidth=3, label='Target Entity'))

        # Add edge direction legend if applicable
        if direction == 'both':
            legend_elements.append(plt.Line2D([0], [0], color='green', linewidth=2,
                                              linestyle='-', label='Outgoing'))
            legend_elements.append(plt.Line2D([0], [0], color='blue', linewidth=2,
                                              linestyle='--', label='Incoming'))

        plt.legend(handles=legend_elements, loc='upper left', title='Entity Types & Directions')

        # Add title with filter information
        filter_info = []
        if direction != 'both':
            filter_info.append(f"dir:{direction}")
        if exclude_types:
            filter_info.append(f"excl_types:{len(exclude_types)}")
        if include_types:
            filter_info.append(f"incl_types:{len(include_types)}")
        if exclude_properties:
            filter_info.append(f"excl_props:{len(exclude_properties)}")
        if include_properties:
            filter_info.append(f"incl_props:{len(include_properties)}")
        if min_connections > 0:
            filter_info.append(f"min_conn:{min_connections}")

        filter_str = f" [{', '.join(filter_info)}]" if filter_info else ""

        title_prefix = "Random " if random_entity else ""
        plt.title(f'{title_prefix}Entity Subgraph: {target_label_str} ({self._format_class_name(target_type)})\n'
                  f'{len(G.nodes())} entities, {len(G.edges())} relationships, max depth: {max_depth}{filter_str}',
                  fontsize=14, fontweight='bold')

        plt.axis('off')
        plt.tight_layout()

        # Save with entity name in filename
        safe_filename = re.sub(r'[^\w\s-]', '', target_label_str).strip()
        safe_filename = ''.join(c for c in safe_filename if ord(c) < 128)[:30]
        if not safe_filename:
            safe_filename = "unknown_entity"

        filename_prefix = "random_" if random_entity else ""
        filter_suffix = "_filtered" if filter_info else ""
        plt.savefig(f"{self.output_dir}/{filename_prefix}entity_subgraph_{safe_filename}{filter_suffix}.png",
                    dpi=300, bbox_inches='tight')

        print(
            f"Entity subgraph saved to {self.output_dir}/{filename_prefix}entity_subgraph_{safe_filename}{filter_suffix}.png")
        print(f"Subgraph contains {len(G.nodes())} entities and {len(G.edges())} relationships")

        # Save subgraph statistics
        with open(f"{self.output_dir}/{filename_prefix}entity_subgraph_{safe_filename}_stats.txt", 'w',
                  encoding='utf-8') as f:
            f.write(f"{'Random ' if random_entity else ''}Entity Subgraph Analysis: {target_label_str}\n")
            f.write(f"Entity Type: {self._format_class_name(target_type)}\n")
            f.write(f"Entity URI: {target_entity}\n")
            f.write(f"Filters Applied: {filter_str}\n\n")

            f.write(f"Subgraph Statistics:\n")
            f.write(f"- Total entities: {len(G.nodes())}\n")
            f.write(f"- Total relationships: {len(G.edges())}\n")
            f.write(f"- Max depth: {max_depth}\n")
            f.write(f"- Direction: {direction}\n\n")

            # Count entities by type
            type_counts = {}
            for n, attr in G.nodes(data=True):
                entity_type = attr.get('type', 'Unknown')
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

            f.write(f"Entity Types in Subgraph:\n")
            for entity_type, count in sorted(type_counts.items()):
                f.write(f"- {entity_type}: {count}\n")

            f.write(f"\nRelationships:\n")
            for u, v, attr in G.edges(data=True):
                relation = attr.get('relation', 'unknown')
                direction_info = attr.get('direction', 'unknown')
                f.write(f"- {u} --[{relation}]({direction_info})--> {v}\n")

    def list_sample_entities(self, entity_type=None, limit=20):
        """
        List sample entities for subgraph visualisation.

        Args:
            entity_type: SMO type to filter by (e.g., "PersonEntry", "MemeEntry")
            limit: Maximum number of entities to return
        """
        print("Sample entities available for subgraph visualisation:")

        entities = []
        type_filter = URIRef(self.SMO + entity_type) if entity_type else None

        for s, p, o in self.g.triples((None, RDF.type, type_filter)):
            if str(s).startswith(str(self.EX)):
                # Check if entity has proper SMO type
                smo_types = [t for t in self.g.objects(s, RDF.type) if str(t).startswith(str(self.SMO))]
                if not smo_types:  # Skip entities without SMO types
                    continue

                # Get label
                labels = list(self.g.objects(s, RDFS.label))
                label = str(labels[0]) if labels else str(s).split('/')[-1]

                # Get type
                if not type_filter:
                    entity_type_str = self._format_class_name(smo_types[0])
                else:
                    entity_type_str = self._format_class_name(type_filter)

                entities.append((label, entity_type_str, str(s)))

        # Sort by label and limit
        entities.sort(key=lambda x: x[0])
        entities = entities[:limit]

        print(f"\nFound {len(entities)} entities" + (f" of type {entity_type}" if entity_type else "") + ":")
        for i, (label, etype, uri) in enumerate(entities, 1):
            print(f"{i:2}. {label} ({etype})")

        print(f"\nTo visualise an entity, use:")
        print(f'analyser.visualise_entity_subgraph(entity_label="Entity Name")')
        print(f'# or')
        print(f'analyser.visualise_entity_subgraph(entity_uri="full_uri")')

        return entities

    def _format_class_name(self, class_uri):
        """Format SMO class name for display."""
        if not isinstance(class_uri, URIRef):
            return str(class_uri)

        uri_str = str(class_uri)
        if uri_str.startswith(str(self.SMO)):
            return f"smo:{uri_str[len(str(self.SMO)):]}"
        elif uri_str.startswith(str(self.SCHEMA)):
            return f"schema:{uri_str[len(str(self.SCHEMA)):]}"
        else:
            return uri_str.split('/')[-1] or uri_str.split('#')[-1]

    def _format_property_name(self, prop_uri):
        """Format property name for display."""
        if not isinstance(prop_uri, URIRef):
            return str(prop_uri)

        uri_str = str(prop_uri)
        if uri_str.startswith(str(self.SMO)):
            return f"smo:{uri_str[len(str(self.SMO)):]}"
        elif uri_str.startswith(str(self.SCHEMA)):
            return f"schema:{uri_str[len(str(self.SCHEMA)):]}"
        elif uri_str.startswith('http://www.w3.org/2000/01/rdf-schema#'):
            return f"rdfs:{uri_str.split('#')[-1]}"
        elif uri_str.startswith('http://xmlns.com/foaf/0.1/'):
            return f"foaf:{uri_str.split('/')[-1]}"
        else:
            return uri_str.split('/')[-1] or uri_str.split('#')[-1]

    def generate_ontology_report(self):
        """Generate comprehensive report for the semiomeme ontology."""
        print("Generating ontology report...")

        report = []
        report.append("# SemioMeme Ontology Analysis Report")
        report.append(f"Analysis performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Graph contains {len(self.g):,} triples\n")

        # Ontology overview
        total_instances = sum(self.entity_counts.values())
        smo_properties = len([p for p in self.property_counts.keys() if str(p).startswith(str(self.SMO))])
        explicit_hierarchies = sum(1 for cls, relations in self.class_relationships.items()
                                   for parent, conf in relations.items() if conf == 1.0)

        report.append("## Ontology Overview")
        report.append(f"- **SMO Classes with Instances**: {len(self.entity_counts)}")
        report.append(f"- **Total Data Instances**: {total_instances:,}")
        report.append(f"- **SMO Properties**: {smo_properties}")
        report.append(f"- **Total Properties**: {len(self.property_counts)}")
        report.append(f"- **Explicit Class Hierarchies**: {explicit_hierarchies}")

        # Class population
        report.append("\n## Class Population")
        for cls, count in list(self.entity_counts.items())[:15]:
            cls_name = self._format_class_name(cls)
            percentage = (count / total_instances) * 100
            report.append(f"- **{cls_name}**: {count:,} instances ({percentage:.1f}%)")

        # Class hierarchy
        report.append("\n## Class Hierarchy")
        if explicit_hierarchies > 0:
            for cls, relations in self.class_relationships.items():
                for parent, conf in relations.items():
                    if conf == 1.0:  # Explicit only
                        cls_name = self._format_class_name(cls)
                        parent_name = self._format_class_name(parent)
                        report.append(f"- {cls_name} subClassOf {parent_name}")
        else:
            report.append("- No explicit class hierarchies found")

        # Property analysis
        report.append("\n## Property Usage Analysis")
        smo_props = [(p, c) for p, c in self.property_counts.items() if str(p).startswith(str(self.SMO))]
        smo_props.sort(key=lambda x: x[1], reverse=True)

        report.append("### Top SMO Properties")
        for prop, count in smo_props[:15]:
            prop_name = self._format_property_name(prop)
            domain = self.property_domains.get(prop)
            range_val = self.property_ranges.get(prop)

            domain_name = self._format_class_name(domain) if domain else "undefined"
            range_name = self._format_class_name(range_val) if range_val else "undefined"

            percentage = (count / len(self.g)) * 100
            report.append(f"- **{prop_name}** ({count:,} uses, {percentage:.2f}% of triples)")
            if domain or range_val:
                report.append(f"  - Domain: {domain_name}, Range: {range_name}")

        # Data quality
        report.append("\n## Data Quality Assessment")
        if self.anomalies:
            anomaly_types = Counter(a['issue'] for a in self.anomalies)
            report.append("### Issues Found")
            for issue, count in anomaly_types.items():
                issue_name = issue.replace('_', ' ').title()
                report.append(f"- **{issue_name}**: {count} instances")

            report.append("\n### Sample Issues")
            for anomaly in self.anomalies[:5]:
                report.append(f"- {anomaly['issue']}: {anomaly['details']}")
        else:
            report.append("- No structural anomalies detected")

        # Statistics summary
        report.append("\n## Statistics Summary")
        report.append(f"- **Triples per Entity**: {len(self.g) / total_instances:.1f}")
        report.append(f"- **Properties per Entity**: {len(self.property_counts) / total_instances:.2f}")
        report.append(f"- **SMO Property Coverage**: {smo_properties / len(self.property_counts) * 100:.1f}%")

        # Most connected entities
        entity_connections = defaultdict(int)
        for s, p, o in self.g:
            if isinstance(s, URIRef) and str(s).startswith(str(self.EX)):
                entity_connections[s] += 1
            if isinstance(o, URIRef) and str(o).startswith(str(self.EX)):
                entity_connections[o] += 1

        if entity_connections:
            top_connected = sorted(entity_connections.items(), key=lambda x: x[1], reverse=True)[:5]
            report.append(f"\n### Most Connected Entities")
            for entity, connections in top_connected:
                entity_name = str(entity).replace(str(self.EX), 'ex:')
                entity_labels = list(self.g.objects(entity, RDFS.label))
                label = str(entity_labels[0]) if entity_labels else entity_name
                report.append(f"- **{label}** ({entity_name}): {connections} connections")

        # Recommendations
        report.append("\n## Recommendations")
        if explicit_hierarchies < 3:
            report.append("1. **Expand Class Hierarchy** - Add more subClassOf relationships for better organisation")

        undefined_domains = len([p for p, c in smo_props if p not in self.property_domains])
        if undefined_domains > 5:
            report.append(f"2. **Define Property Domains** - {undefined_domains} SMO properties lack explicit domains")

        singleton_props = len([p for p, c in smo_props if c == 1])
        if singleton_props > 10:
            report.append(f"3. **Review Singleton Properties** - {singleton_props} SMO properties used only once")

        # Save report
        with open(f"{self.output_dir}/ontology_analysis_report.md", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"Analysis report saved to {self.output_dir}/ontology_analysis_report.md")

    def generate_summary_statistics(self):
        """
        Generate comprehensive summary statistics for the knowledge graph.
        Based on standard KG evaluation metrics from literature.
        """
        print("Generating comprehensive knowledge graph statistics...")

        statistics = {}

        # === BASIC GRAPH STATISTICS ===
        statistics['basic'] = {
            'total_triples': len(self.g),
            'total_unique_subjects': len(set(s for s, p, o in self.g)),
            'total_unique_predicates': len(set(p for s, p, o in self.g)),
            'total_unique_objects': len(set(o for s, p, o in self.g)),
            'total_unique_entities': len(set(list(set(s for s, p, o in self.g)) +
                                             [o for s, p, o in self.g if isinstance(o, URIRef)])),
            'total_literals': sum(1 for s, p, o in self.g if isinstance(o, Literal))
        }

        # === ONTOLOGY STRUCTURE STATISTICS ===
        smo_classes = len(self.entity_counts)
        smo_properties = len([p for p in self.property_counts.keys() if str(p).startswith(str(self.SMO))])

        statistics['ontology'] = {
            'smo_classes_with_instances': smo_classes,
            'total_class_instances': sum(self.entity_counts.values()),
            'smo_properties': smo_properties,
            'total_properties': len(self.property_counts),
            'explicit_class_hierarchies': sum(1 for cls, relations in self.class_relationships.items()
                                              for parent, conf in relations.items() if conf == 1.0),
            'ontology_coverage': (smo_properties / len(self.property_counts)) * 100 if self.property_counts else 0
        }

        # === CONNECTIVITY STATISTICS ===
        # Calculate node degrees for EX entities only
        entity_outgoing = defaultdict(int)
        entity_incoming = defaultdict(int)
        entity_total = defaultdict(int)

        ex_entities = set()
        for s, p, o in self.g:
            if isinstance(s, URIRef) and str(s).startswith(str(self.EX)):
                ex_entities.add(s)
                entity_outgoing[s] += 1
                entity_total[s] += 1
            if isinstance(o, URIRef) and str(o).startswith(str(self.EX)):
                ex_entities.add(o)
                entity_incoming[o] += 1
                entity_total[o] += 1

        if ex_entities:
            outgoing_degrees = [entity_outgoing[e] for e in ex_entities]
            incoming_degrees = [entity_incoming[e] for e in ex_entities]
            total_degrees = [entity_total[e] for e in ex_entities]

            statistics['connectivity'] = {
                'total_ex_entities': len(ex_entities),
                'avg_outgoing_degree': sum(outgoing_degrees) / len(outgoing_degrees),
                'avg_incoming_degree': sum(incoming_degrees) / len(incoming_degrees),
                'avg_total_degree': sum(total_degrees) / len(total_degrees),
                'max_outgoing_degree': max(outgoing_degrees) if outgoing_degrees else 0,
                'max_incoming_degree': max(incoming_degrees) if incoming_degrees else 0,
                'max_total_degree': max(total_degrees) if total_degrees else 0,
                'min_outgoing_degree': min(outgoing_degrees) if outgoing_degrees else 0,
                'min_incoming_degree': min(incoming_degrees) if incoming_degrees else 0,
                'isolated_entities': sum(1 for e in ex_entities if entity_total[e] == 0)
            }
        else:
            statistics['connectivity'] = {
                'total_ex_entities': 0,
                'avg_outgoing_degree': 0,
                'avg_incoming_degree': 0,
                'avg_total_degree': 0,
                'max_outgoing_degree': 0,
                'max_incoming_degree': 0,
                'max_total_degree': 0,
                'min_outgoing_degree': 0,
                'min_incoming_degree': 0,
                'isolated_entities': 0
            }

        # === DENSITY AND SPARSITY ===
        n_entities = len(ex_entities)
        max_possible_edges = n_entities * (n_entities - 1) if n_entities > 1 else 0
        actual_edges = sum(1 for s, p, o in self.g
                           if isinstance(s, URIRef) and isinstance(o, URIRef)
                           and str(s).startswith(str(self.EX)) and str(o).startswith(str(self.EX)))

        statistics['density'] = {
            'graph_density': (actual_edges / max_possible_edges) * 100 if max_possible_edges > 0 else 0,
            'sparsity': ((
                                     max_possible_edges - actual_edges) / max_possible_edges) * 100 if max_possible_edges > 0 else 100,
            'edge_entity_ratio': actual_edges / n_entities if n_entities > 0 else 0,
            'triple_entity_ratio': len(self.g) / n_entities if n_entities > 0 else 0
        }

        # === PROPERTY USAGE STATISTICS ===
        property_usage = list(self.property_counts.values())
        if property_usage:
            statistics['property_usage'] = {
                'most_used_property_count': max(property_usage),
                'least_used_property_count': min(property_usage),
                'avg_property_usage': sum(property_usage) / len(property_usage),
                'median_property_usage': sorted(property_usage)[len(property_usage) // 2],
                'singleton_properties': sum(1 for count in property_usage if count == 1),
                'heavily_used_properties': sum(1 for count in property_usage if count > 1000)
            }
        else:
            statistics['property_usage'] = {}

        # === CLASS DISTRIBUTION STATISTICS ===
        class_populations = list(self.entity_counts.values())
        if class_populations:
            avg_class_size = sum(class_populations) / len(class_populations)
            statistics['class_distribution'] = {
                'largest_class_size': max(class_populations),
                'smallest_class_size': min(class_populations),
                'avg_class_size': avg_class_size,
                'median_class_size': sorted(class_populations)[len(class_populations) // 2],
                'class_size_std_dev': (sum((x - avg_class_size) ** 2
                                           for x in class_populations) / len(class_populations)) ** 0.5,
                'empty_classes': 0,  # We only count classes with instances
                'classes_over_1000': sum(1 for pop in class_populations if pop > 1000)
            }
        else:
            statistics['class_distribution'] = {}

        # === ENRICHMENT STATISTICS ===
        wikidata_linked = 0
        schema_enriched = 0
        multi_typed = 0

        for s, p, o in self.g:
            if p == OWL.sameAs and str(o).startswith('http://www.wikidata.org/'):
                wikidata_linked += 1
            elif p == OWL.sameAs:
                schema_enriched += 1

        # Count entities with multiple SMO types
        entity_type_counts = defaultdict(int)
        for s, p, o in self.g.triples((None, RDF.type, None)):
            if str(o).startswith(str(self.SMO)):
                entity_type_counts[s] += 1

        multi_typed = sum(1 for count in entity_type_counts.values() if count > 1)

        statistics['enrichment'] = {
            'wikidata_linked_entities': wikidata_linked,
            'external_linked_entities': schema_enriched,
            'total_enriched_entities': wikidata_linked + schema_enriched,
            'enrichment_rate': ((wikidata_linked + schema_enriched) / n_entities) * 100 if n_entities > 0 else 0,
            'multi_typed_entities': multi_typed,
            'multi_type_rate': (multi_typed / n_entities) * 100 if n_entities > 0 else 0
        }

        # === DATA QUALITY INDICATORS ===
        untyped_entities = 0
        for s in ex_entities:
            smo_types = [t for t in self.g.objects(s, RDF.type) if str(t).startswith(str(self.SMO))]
            if not smo_types:
                untyped_entities += 1

        statistics['quality'] = {
            'untyped_entities': untyped_entities,
            'typed_entity_rate': ((n_entities - untyped_entities) / n_entities) * 100 if n_entities > 0 else 0,
            'total_anomalies': len(self.anomalies),
            'data_completeness': 100 - ((untyped_entities / n_entities) * 100) if n_entities > 0 else 0
        }

        # === TEMPORAL COVERAGE ===
        temporal_properties = [URIRef(str(self.SMO) + prop) for prop in
                               ['yearCreated', 'dateCreated', 'temporalDescription']]
        temporal_triples = sum(1 for s, p, o in self.g if p in temporal_properties)

        statistics['temporal'] = {
            'entities_with_temporal_data': len(set(s for s, p, o in self.g if p in temporal_properties)),
            'total_temporal_triples': temporal_triples,
            'temporal_coverage': (len(set(
                s for s, p, o in self.g if p in temporal_properties)) / n_entities) * 100 if n_entities > 0 else 0
        }

        # === SUMMARY SCORES ===
        # Calculate composite quality scores
        completeness_score = statistics['quality']['data_completeness']
        connectivity_score = min(100, statistics['connectivity']['avg_total_degree'] * 10)
        enrichment_score = statistics['enrichment']['enrichment_rate']

        statistics['summary_scores'] = {
            'data_completeness_score': completeness_score,
            'connectivity_score': connectivity_score,
            'enrichment_score': enrichment_score,
            'overall_quality_score': (completeness_score + connectivity_score + enrichment_score) / 3
        }

        # Save detailed statistics
        self._save_statistics_to_files(statistics)

        # Print summary
        self._print_statistics_summary(statistics)

        return statistics

    def _save_statistics_to_files(self, statistics):
        """Save statistics to CSV file only."""

        # Save as CSV for spreadsheet analysis
        with open(f"{self.output_dir}/kg_statistics.csv", 'w', encoding='utf-8') as f:
            f.write("Category,Metric,Value,Description\n")
            for category, metrics in statistics.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        description = self._get_metric_description(category, metric)
                        f.write(f'"{category}","{metric}","{value}","{description}"\n')

    def _convert_uris_to_strings(self, obj):
        """Recursively convert URIRef objects to strings for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_uris_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_uris_to_strings(item) for item in obj]
        elif isinstance(obj, URIRef):
            return str(obj)
        else:
            return obj

    def _get_metric_description(self, category, metric):
        """Get human-readable description for metrics."""
        descriptions = {
            'basic': {
                'total_triples': 'Total number of RDF triples in the graph',
                'total_unique_subjects': 'Number of unique subject entities',
                'total_unique_predicates': 'Number of unique properties/predicates',
                'total_unique_objects': 'Number of unique object entities and literals',
                'total_unique_entities': 'Total unique entities (subjects + object URIs)',
                'total_literals': 'Number of literal values (strings, numbers, dates)'
            },
            'ontology': {
                'smo_classes_with_instances': 'SMO ontology classes that have at least one instance',
                'total_class_instances': 'Total instances across all SMO classes',
                'smo_properties': 'Number of SMO ontology properties',
                'total_properties': 'Total properties including external vocabularies',
                'explicit_class_hierarchies': 'Number of explicit subClassOf relationships',
                'ontology_coverage': 'Percentage of properties that are SMO-defined'
            },
            'connectivity': {
                'avg_outgoing_degree': 'Average number of outgoing connections per entity',
                'avg_incoming_degree': 'Average number of incoming connections per entity',
                'max_total_degree': 'Highest number of total connections for any entity',
                'isolated_entities': 'Entities with no connections'
            },
            'density': {
                'graph_density': 'Percentage of possible edges that actually exist',
                'sparsity': 'Percentage of possible edges that do not exist',
                'edge_entity_ratio': 'Average number of edges per entity',
                'triple_entity_ratio': 'Average number of triples per entity'
            },
            'enrichment': {
                'wikidata_linked_entities': 'Entities linked to Wikidata via owl:sameAs',
                'enrichment_rate': 'Percentage of entities with external links',
                'multi_typed_entities': 'Entities with multiple SMO type assertions'
            },
            'quality': {
                'untyped_entities': 'Entities without any SMO type classification',
                'typed_entity_rate': 'Percentage of entities with SMO types',
                'data_completeness': 'Overall data completeness score'
            }
        }

        return descriptions.get(category, {}).get(metric, 'Knowledge graph metric')

    def _print_statistics_summary(self, statistics):
        """Print a concise summary of key statistics."""
        print("Knowledge graph statistics generated successfully")
        print(f"CSV saved to: {self.output_dir}/kg_statistics.csv")

    def analyse_connectivity_patterns(self, top_n=20, save_charts=True):
        """
        Analyse connectivity patterns in the knowledge graph to identify highly connected entities
        and validate cross-referencing between memes, series, tags, and people.
        """
        print("Analysing connectivity patterns...")

        connectivity_data = {}

        # === ANALYSE BY PROPERTY TYPE ===
        property_connections = defaultdict(lambda: defaultdict(int))

        # Count connections for each property type
        for s, p, o in self.g:
            if isinstance(o, URIRef) and str(o).startswith(str(self.EX)):
                prop_name = self._format_property_name(p)

                # Get labels for better readability
                s_labels = list(self.g.objects(s, RDFS.label))
                o_labels = list(self.g.objects(o, RDFS.label))
                s_label = str(s_labels[0]) if s_labels else str(s).split('/')[-1]
                o_label = str(o_labels[0]) if o_labels else str(o).split('/')[-1]

                property_connections[prop_name][o_label] += 1

        # === ANALYSE SERIES CONNECTIONS ===
        series_connections = {}
        part_of_series_prop = URIRef(str(self.SMO) + 'partOfSeries')

        for s, p, o in self.g.triples((None, part_of_series_prop, None)):
            if isinstance(o, URIRef):
                o_labels = list(self.g.objects(o, RDFS.label))
                series_name = str(o_labels[0]) if o_labels else str(o).split('/')[-1]
                series_connections[series_name] = series_connections.get(series_name, 0) + 1

        # === ANALYSE TAG CONNECTIONS ===
        tag_connections = {}
        has_tag_prop = URIRef(str(self.SMO) + 'hasTag')

        for s, p, o in self.g.triples((None, has_tag_prop, None)):
            if isinstance(o, URIRef):
                o_labels = list(self.g.objects(o, RDFS.label))
                tag_name = str(o_labels[0]) if o_labels else str(o).split('/')[-1]
                tag_connections[tag_name] = tag_connections.get(tag_name, 0) + 1

        # === ANALYSE PERSON MENTIONS ===
        person_mentions = {}
        mentions_prop = URIRef(str(self.SMO) + 'mentions')

        for s, p, o in self.g.triples((None, mentions_prop, None)):
            if isinstance(o, URIRef):
                # Check if this is a person
                o_types = list(self.g.objects(o, RDF.type))
                if any('PersonEntry' in str(t) for t in o_types):
                    o_labels = list(self.g.objects(o, RDFS.label))
                    person_name = str(o_labels[0]) if o_labels else str(o).split('/')[-1]
                    person_mentions[person_name] = person_mentions.get(person_name, 0) + 1

        meme_format_connections = {}
        has_meme_format_prop = URIRef(str(self.SMO) + 'hasMemeFormat')

        for s, p, o in self.g.triples((None, has_meme_format_prop, None)):
            if isinstance(o, URIRef):
                o_labels = list(self.g.objects(o, RDFS.label))
                format_name = str(o_labels[0]) if o_labels else str(o).split('/')[-1]
                meme_format_connections[format_name] = meme_format_connections.get(format_name, 0) + 1

        # === OVERALL CONNECTION ANALYSIS ===
        # Count total connections per entity (incoming + outgoing)
        entity_total_connections = defaultdict(int)
        entity_labels_map = {}

        for s, p, o in self.g:
            if isinstance(s, URIRef) and str(s).startswith(str(self.EX)):
                entity_total_connections[s] += 1
                if s not in entity_labels_map:
                    s_labels = list(self.g.objects(s, RDFS.label))
                    entity_labels_map[s] = str(s_labels[0]) if s_labels else str(s).split('/')[-1]

            if isinstance(o, URIRef) and str(o).startswith(str(self.EX)):
                entity_total_connections[o] += 1
                if o not in entity_labels_map:
                    o_labels = list(self.g.objects(o, RDFS.label))
                    entity_labels_map[o] = str(o_labels[0]) if o_labels else str(o).split('/')[-1]

        # Convert to name-based dictionary for readability
        entity_connections_by_name = {
            entity_labels_map[entity]: count
            for entity, count in entity_total_connections.items()
        }

        # === COMPILE RESULTS ===
        connectivity_data = {
            'series_connections': dict(sorted(series_connections.items(), key=lambda x: x[1], reverse=True)[:top_n]),
            'tag_connections': dict(sorted(tag_connections.items(), key=lambda x: x[1], reverse=True)[:top_n]),
            'person_mentions': dict(sorted(person_mentions.items(), key=lambda x: x[1], reverse=True)[:top_n]),
            'meme_format_connections': dict(
                sorted(meme_format_connections.items(), key=lambda x: x[1], reverse=True)[:top_n]),  # ADD THIS
            'most_connected_entities': dict(
                sorted(entity_connections_by_name.items(), key=lambda x: x[1], reverse=True)[:top_n]),
            'property_usage_summary': {prop: len(connections) for prop, connections in property_connections.items()}
        }

        # === SAVE DETAILED CSV ===
        self._save_connectivity_csv(connectivity_data, property_connections, top_n)

        # === CREATE VISUALISATIONS ===
        if save_charts:
            self._create_connectivity_charts(connectivity_data, top_n)

        # === PRINT SUMMARY ===
        self._print_connectivity_summary(connectivity_data)

        return connectivity_data

    def _save_connectivity_csv(self, connectivity_data, property_connections, top_n):
        """Save detailed connectivity analysis to CSV files."""

        # Overall connectivity summary
        with open(f"{self.output_dir}/connectivity_analysis.csv", 'w', encoding='utf-8') as f:
            f.write("Entity_Type,Entity_Name,Connection_Count,Entity_Type_Category\n")

            # Series
            for series, count in connectivity_data['series_connections'].items():
                f.write(f'"Series","{series}",{count},"Content Grouping"\n')

            # Tags
            for tag, count in connectivity_data['tag_connections'].items():
                f.write(f'"Tag","{tag}",{count},"Classification"\n')

            # People
            for person, count in connectivity_data['person_mentions'].items():
                f.write(f'"Person","{person}",{count},"Individual"\n')

            # Most connected overall
            for entity, count in connectivity_data['most_connected_entities'].items():
                f.write(f'"Entity","{entity}",{count},"General"\n')

        # Property-specific connections
        with open(f"{self.output_dir}/property_connections.csv", 'w', encoding='utf-8') as f:
            f.write("Property,Connected_Entity,Connection_Count\n")

            for prop, connections in property_connections.items():
                sorted_connections = sorted(connections.items(), key=lambda x: x[1], reverse=True)[:top_n]
                for entity, count in sorted_connections:
                    f.write(f'"{prop}","{entity}",{count}\n')

    def _create_connectivity_charts(self, connectivity_data, top_n):
        """Create visualisation charts for connectivity patterns."""

        # Create a 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # 1. Series Connections
        if connectivity_data['series_connections']:
            series_names = list(connectivity_data['series_connections'].keys())[:15]
            series_counts = list(connectivity_data['series_connections'].values())[:15]

            ax1.barh(range(len(series_names)), series_counts, color='lightblue', edgecolor='darkblue')
            ax1.set_yticks(range(len(series_names)))
            ax1.set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in series_names])
            ax1.set_xlabel('Number of Connected Memes')
            ax1.set_title('Top Series by Number of Connected Memes')
            ax1.grid(axis='x', alpha=0.3)

        # 2. Tag Connections
        if connectivity_data['tag_connections']:
            tag_names = list(connectivity_data['tag_connections'].keys())[:15]
            tag_counts = list(connectivity_data['tag_connections'].values())[:15]

            ax2.barh(range(len(tag_names)), tag_counts, color='lightgreen', edgecolor='darkgreen')
            ax2.set_yticks(range(len(tag_names)))
            ax2.set_yticklabels([name[:25] + '...' if len(name) > 25 else name for name in tag_names])
            ax2.set_xlabel('Number of Tagged Entities')
            ax2.set_title('Top Tags by Usage Frequency')
            ax2.grid(axis='x', alpha=0.3)

        # 3. Person Mentions
        if connectivity_data['person_mentions']:
            person_names = list(connectivity_data['person_mentions'].keys())[:15]
            person_counts = list(connectivity_data['person_mentions'].values())[:15]

            ax3.barh(range(len(person_names)), person_counts, color='lightcoral', edgecolor='darkred')
            ax3.set_yticks(range(len(person_names)))
            ax3.set_yticklabels([name[:25] + '...' if len(name) > 25 else name for name in person_names])
            ax3.set_xlabel('Number of Meme Mentions')
            ax3.set_title('Most Mentioned People Across Memes')
            ax3.grid(axis='x', alpha=0.3)

        # 4. Most Connected Entities Overall
        if connectivity_data['most_connected_entities']:
            entity_names = list(connectivity_data['most_connected_entities'].keys())[:15]
            entity_counts = list(connectivity_data['most_connected_entities'].values())[:15]

            ax4.barh(range(len(entity_names)), entity_counts, color='lightyellow', edgecolor='orange')
            ax4.set_yticks(range(len(entity_names)))
            ax4.set_yticklabels([name[:25] + '...' if len(name) > 25 else name for name in entity_names])
            ax4.set_xlabel('Total Connections')
            ax4.set_title('Most Connected Entities (All Types)')
            ax4.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/connectivity_analysis.png", dpi=300, bbox_inches='tight')
        print(f"Connectivity charts saved to {self.output_dir}/connectivity_analysis.png")

    def _print_connectivity_summary(self, connectivity_data):
        """Print a summary of connectivity analysis findings."""

        print("\nConnectivity Analysis Summary:")
        print("=" * 50)

        # Series analysis
        if connectivity_data['series_connections']:
            top_series = list(connectivity_data['series_connections'].items())[0]
            print(f"Top Series: '{top_series[0]}' with {top_series[1]} connected memes")
            series_over_10 = sum(1 for count in connectivity_data['series_connections'].values() if count >= 10)
            print(f"Series with 10+ memes: {series_over_10}")

        # Tag analysis
        if connectivity_data['tag_connections']:
            top_tag = list(connectivity_data['tag_connections'].items())[0]
            print(f"Top Tag: '{top_tag[0]}' used {top_tag[1]} times")
            tags_over_100 = sum(1 for count in connectivity_data['tag_connections'].values() if count >= 100)
            print(f"Tags used 100+ times: {tags_over_100}")

        # Person analysis
        if connectivity_data['person_mentions']:
            top_person = list(connectivity_data['person_mentions'].items())[0]
            print(f"Most Mentioned Person: '{top_person[0]}' in {top_person[1]} memes")
            people_in_multiple = sum(1 for count in connectivity_data['person_mentions'].values() if count >= 2)
            print(f"People mentioned in 2+ memes: {people_in_multiple}")

        # Overall connectivity
        if connectivity_data['most_connected_entities']:
            most_connected = list(connectivity_data['most_connected_entities'].items())[0]
            print(f"Most Connected Entity: '{most_connected[0]}' with {most_connected[1]} total connections")

        print(f"\nDetailed results saved to:")
        print(f"  - {self.output_dir}/connectivity_analysis.csv")
        print(f"  - {self.output_dir}/property_connections.csv")
        print("=" * 50)

    def run_all(self, fix_anomalies=False, generate_stats=True, analyse_connectivity=True):
        """Run complete analysis suite with optional statistics and connectivity analysis."""
        self.analyse_graph()
        self.visualise_graph_structure()

        if generate_stats:
            self.generate_summary_statistics()

        if analyse_connectivity:
            self.analyse_connectivity_patterns()

        self.generate_ontology_report()

        if fix_anomalies:
            print("Note: Anomaly fixing not implemented - manual review recommended")

        print(f"\nAnalysis complete. Results saved to {self.output_dir}/")
        print(f"Key findings:")
        print(f"  - {len(self.entity_counts)} SMO classes with {sum(self.entity_counts.values()):,} total instances")
        print(f"  - {len([p for p in self.property_counts.keys() if str(p).startswith(str(self.SMO))])} SMO properties")
        print(f"  - {len(self.anomalies)} structural issues detected")

        print(f"\nEntity visualisation options:")
        print(f"  analyser.list_sample_entities()  # Show available entities")
        print(f"  analyser.visualise_entity_subgraph(entity_label='Entity Name')")
        print(f"  analyser.visualise_entity_subgraph(random_entity=True)  # Random entity")
        print(f"  analyser.visualise_entity_subgraph(random_entity=True, entity_type_filter='PersonEntry')")

        self.output_manager.cleanup_old_sessions()
        print(f"Analysis complete. Results in: {self.output_dir}")

    def run_entity_analysis(self, entity_label=None, random_entity=False, entity_type_filter=None):
        """Quick method to analyse a single entity."""
        if random_entity:
            print(f"Analysing random entity" + (f" of type {entity_type_filter}" if entity_type_filter else ""))
            self.visualise_entity_subgraph(random_entity=True, entity_type_filter=entity_type_filter)
        elif entity_label:
            print(f"Analysing entity: {entity_label}")
            self.visualise_entity_subgraph(entity_label=entity_label)
        else:
            print("Please specify either entity_label or set random_entity=True")



