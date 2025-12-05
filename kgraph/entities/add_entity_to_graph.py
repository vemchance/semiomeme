from rdflib import Literal, URIRef, RDF, RDFS

def add_entity_to_graph(g, namespace, entity_uri, entity_type_class, entity_registry, entity_types,
                        entity_resolver, ontology_mapper, label=None, related_fields=None):
    """
    Add an entity to the graphs with proper ontology typing.

    Args:
        entity_type_class: URIRef to ontology class (e.g., SMO.Tag, SMO.Status)
    """
    entity_node = URIRef(namespace[entity_uri])

    if entity_uri not in entity_registry:
        g.add((entity_node, RDF.type, entity_type_class))
        entity_registry[entity_uri] = {'uri': entity_node, 'type': entity_type_class}
        entity_types.add(entity_type_class)

        if label:
            g.add((entity_node, RDFS.label, Literal(label)))

    if entity_uri not in entity_registry:
        ontology_mapper.map_entity_to_ontologies(
            g, entity_uri, entity_type_class, namespace, label or entity_uri
        )

    return entity_node