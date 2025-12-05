from rdflib import Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS, DCTERMS


class OntologyMapper:
    """
    Ontology mapper that provides hierarchical mappings and direct entity enrichment.
    """

    def __init__(self, entity_linker):
        """
        Initialize the OntologyMapper.

        Args:
            entity_linker: An instance of WikidataEntityLinker
        """
        self.entity_linker = entity_linker

    def bind_namespaces(self, graph):
        """Bind common ontology namespaces to the graphs."""
        # Bind common ontologies
        graph.bind("skos", SKOS)
        graph.bind("dcterms", DCTERMS)
        graph.bind("xsd", XSD)
        graph.bind("owl", OWL)
        graph.bind("rdfs", RDFS)

        # Define and bind additional namespaces
        SCHEMA = Namespace("http://schema.org/")
        DBR = Namespace("http://dbpedia.org/resource/")
        DBO = Namespace("http://dbpedia.org/ontology/")

        graph.bind("schema", SCHEMA)
        graph.bind("dbr", DBR)
        graph.bind("dbo", DBO)

    def define_class_hierarchies(self, graph):
        """
        Define proper ontological structure with separate namespace for classes.
        Follows Semantic Web best practices for vocabulary design.
        """
        # Define separate namespaces
        SMO = Namespace("http://semiomeme.org/ontology/")  # Ontology classes/properties
        EX = Namespace("http://semiomeme.org/data/")  # Data instances
        SCHEMA = Namespace("http://schema.org/")
        FOAF = Namespace("http://xmlns.com/foaf/0.1/")
        DCTERMS = Namespace("http://purl.org/dc/terms/")

        # Bind namespaces
        graph.bind("smo", SMO)
        graph.bind("ex", EX)
        graph.bind("schema", SCHEMA)
        graph.bind("foaf", FOAF)
        graph.bind("dcterms", DCTERMS)

        # === Core Ontology Classes ===

        # Root class for all KYM documentation entries
        kym_entry = SMO.KYMEntry
        graph.add((kym_entry, RDF.type, OWL.Class))
        graph.add((kym_entry, RDFS.subClassOf, SCHEMA.CreativeWork))
        graph.add((kym_entry, RDFS.subClassOf, FOAF.Document))  # It documents something
        graph.add((kym_entry, RDFS.label, Literal("KYM Documentation Entry")))
        graph.add((kym_entry, RDFS.comment,
                   Literal("A documentation entry from Know Your Meme about internet cultural phenomena")))

        # === KYM Entry Types (all are forms of documentation) ===

        # Meme documentation
        meme_entry = SMO.MemeEntry
        graph.add((meme_entry, RDF.type, OWL.Class))
        graph.add((meme_entry, RDFS.subClassOf, kym_entry))
        graph.add((meme_entry, RDFS.label, Literal("Meme Entry")))
        graph.add((meme_entry, RDFS.comment,
                   Literal("KYM documentation about a meme or viral content")))

        # Person documentation
        person_entry = SMO.PersonEntry
        graph.add((person_entry, RDF.type, OWL.Class))
        graph.add((person_entry, RDFS.subClassOf, kym_entry))
        graph.add((person_entry, RDFS.subClassOf, FOAF.PersonalProfileDocument))
        graph.add((person_entry, RDFS.label, Literal("Person Entry")))
        graph.add((person_entry, RDFS.comment,
                   Literal("KYM documentation about a person relevant to internet culture")))

        # Event documentation
        event_entry = SMO.EventEntry
        graph.add((event_entry, RDF.type, OWL.Class))
        graph.add((event_entry, RDFS.subClassOf, kym_entry))
        graph.add((event_entry, RDFS.label, Literal("Event Entry")))
        graph.add((event_entry, RDFS.comment,
                   Literal("KYM documentation about an event relevant to internet culture")))

        # Site documentation
        site_entry = SMO.SiteEntry
        graph.add((site_entry, RDF.type, OWL.Class))
        graph.add((site_entry, RDFS.subClassOf, kym_entry))
        graph.add((site_entry, RDFS.label, Literal("Site Entry")))
        graph.add((site_entry, RDFS.comment,
                   Literal("KYM documentation about a website or online platform")))

        # Subculture documentation
        subculture_entry = SMO.SubcultureEntry
        graph.add((subculture_entry, RDF.type, OWL.Class))
        graph.add((subculture_entry, RDFS.subClassOf, kym_entry))
        graph.add((subculture_entry, RDFS.label, Literal("Subculture Entry")))
        graph.add((subculture_entry, RDFS.comment,
                   Literal("KYM documentation about a subculture or community")))


        # Generic entity for REBEL-extracted entities that don't fit other categories
        entity_class = SMO.Entity
        graph.add((entity_class, RDF.type, OWL.Class))
        graph.add((entity_class, RDFS.subClassOf, kym_entry))
        graph.add((entity_class, RDFS.label, Literal("Generic Entity Entry")))
        graph.add((entity_class, RDFS.comment,
                   Literal("KYM documentation about an entity that doesn't fit other specific categories")))

        # Individual meme instance class
        meme_instance = SMO.MemeInstance
        graph.add((meme_instance, RDF.type, OWL.Class))
        graph.add((meme_instance, RDFS.subClassOf, meme_entry))
        graph.add((meme_instance, RDFS.label, Literal("Meme Instance")))
        graph.add((meme_instance, RDFS.comment,
                   Literal("Individual meme file belonging to a KYM entry")))

        # === KYM Metadata Classes ===

        # Root metadata class
        kym_metadata = SMO.KYMMetadata
        graph.add((kym_metadata, RDF.type, OWL.Class))
        graph.add((kym_metadata, RDFS.subClassOf, SCHEMA.Intangible))
        graph.add((kym_metadata, RDFS.label, Literal("KYM Metadata")))
        graph.add((kym_metadata, RDFS.comment,
                   Literal("Metadata concepts used for KYM classification and organisation")))

        # Add provenance properties
        extraction_prop = SMO.extractedBy
        graph.add((extraction_prop, RDF.type, OWL.DatatypeProperty))
        graph.add((extraction_prop, RDFS.domain, RDF.Property))
        graph.add((extraction_prop, RDFS.range, XSD.string))
        graph.add((extraction_prop, RDFS.label, Literal("extracted by")))
        graph.add((extraction_prop, RDFS.comment,
                   Literal("Indicates the extraction method used for this predicate")))



        # Specific metadata types
        metadata_classes = [
            ("MemeType", "Format styles like Image Macro, Exploitable, Remix"),
            ("Tag", "Tag for categorising content"),
            ("Status", "Editorial status of KYM entry"),
            ("Badge", "Special badge assigned to entry"),
            ("Series", "Series or collection grouping")
        ]

        for class_name, description in metadata_classes:
            class_uri = getattr(SMO, class_name)
            graph.add((class_uri, RDF.type, OWL.Class))
            graph.add((class_uri, RDFS.subClassOf, kym_metadata))
            graph.add((class_uri, RDFS.label, Literal(class_name)))
            graph.add((class_uri, RDFS.comment, Literal(description)))

        # === Properties ===

        # Core property linking KYM entries to real-world entities
        documents_prop = SMO.documents
        graph.add((documents_prop, RDF.type, OWL.ObjectProperty))
        graph.add((documents_prop, RDFS.domain, kym_entry))
        graph.add((documents_prop, RDFS.range, SCHEMA.Thing))
        graph.add((documents_prop, RDFS.label, Literal("documents")))
        graph.add((documents_prop, RDFS.comment,
                   Literal("Links a KYM entry to the real-world entity it documents")))

        # KYM-specific properties
        kym_properties = [
            ("hasStatus", kym_entry, SMO.Status, "Editorial status of the entry"),
            ("hasBadge", kym_entry, SMO.Badge, "Special badge assigned to entry"),
            ("hasTag", kym_entry, SMO.Tag, "Tag associated with the entry"),
            ("hasMemeType", SMO.KYMEntry, SMO.MemeType, "Format classification for memes (Image Macro, Exploitable, etc.)"),
            ("partOfSeries", kym_entry, SMO.Series, "Series this entry belongs to"),
            ("mentions", kym_entry, SCHEMA.Thing, "Entity mentioned in the entry"),
            ("belongsTo", SMO.MemeInstance, kym_entry, "KYM entry this instance belongs to")
        ]

        for prop_name, domain, range_cls, description in kym_properties:
            prop_uri = getattr(SMO, prop_name)
            graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
            graph.add((prop_uri, RDFS.domain, domain))
            graph.add((prop_uri, RDFS.range, range_cls))
            graph.add((prop_uri, RDFS.label, Literal(prop_name.lower().replace('has', '').replace('of', ' of'))))
            graph.add((prop_uri, RDFS.comment, Literal(description)))

        # Data properties for KYM metadata
        data_properties = [
            ("yearCreated", kym_entry, XSD.gYear, "Year the documented phenomenon was created"),
            ("originDescription", kym_entry, XSD.string, "Description of where/how it originated"),
            ("dateOfBirth", SMO.PersonEntry, XSD.date, "Birth date of a person"),
            ("dateCreated", kym_entry, XSD.date, "Specific date when created"),
            ("temporalDescription", kym_entry, XSD.string, "Temporal description when precise date unknown"),
            ("kymID", kym_entry, XSD.string, "Unique identifier from KnowYourMeme.com and ID for Corpus Bridge"),

            # === ADD THESE NEW PROPERTIES ===

            # Engagement metrics
            ("viewCount", kym_entry, XSD.integer, "Number of views on KnowYourMeme"),
            ("videoCount", kym_entry, XSD.integer, "Number of videos associated with entry"),
            ("photoCount", kym_entry, XSD.integer, "Number of photos/images associated with entry"),
            ("commentCount", kym_entry, XSD.integer, "Number of comments on entry"),

            # Media and references
            ("mainImageURL", kym_entry, XSD.string, "URL of the main image for this entry"),
            ("kymURL", kym_entry, XSD.string, "Original KnowYourMeme.com URL"),
            ("externalReferenceCount", kym_entry, XSD.integer, "Number of external references"),

            # Additional metadata
            ("region", kym_entry, XSD.string, "Geographic region associated with meme"),
            ("metaDescription", kym_entry, XSD.string, "Meta description from KYM page"),

            # Text content properties
            ("aboutText", kym_entry, XSD.string, "About section text - not currently used"),
            ("originText", kym_entry, XSD.string, "Origin section text - not currently used"),
            ("spreadText", kym_entry, XSD.string, "Spread section text - not currently used"),

            # Corpus layer properties (existing)
            ("faissVisionIndex", SMO.MemeInstance, XSD.string, "Reference to image embedding in FAISS"),
            ("faissTextIndex", SMO.MemeInstance, XSD.string, "Reference to text embedding in FAISS"),
            ("hasImagePath", SMO.MemeInstance, XSD.string, "File name of meme image"),
            ("hasModality", SMO.MemeInstance, XSD.string, "Modality extracted from the instance"),
            ("datasetType", SMO.MemeInstance, XSD.string, "Whether the meme instance is confirmed or unconfirmed"),
            ("hasOCRText", SMO.MemeInstance, XSD.string, "Extracted text from meme image via OCR"),
            ("popularity", SMO.MemeInstance, XSD.integer, "Popularity score based on number of instances"),
        ]

        for prop_name, domain, range_type, description in data_properties:
            prop_uri = getattr(SMO, prop_name)
            graph.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            graph.add((prop_uri, RDFS.domain, domain))
            graph.add((prop_uri, RDFS.range, range_type))
            graph.add((prop_uri, RDFS.label, Literal(prop_name.lower())))
            graph.add((prop_uri, RDFS.comment, Literal(description)))

    def map_entity_to_ontologies(self, graph, entity_uri, entity_type_class, namespace, label=None):
        """
        Map entities to ontologies using BOTH hierarchical approach AND direct enrichment.

        Args:
            graph: The RDF graphs to add mappings to
            entity_uri: The URI string of the entity in the local namespace
            entity_type_class: URIRef to the ontology class (e.g., SMO.PersonEntry, SMO.MemeEntry)
            namespace: The local data namespace (EX)
            label: Optional label for the entity
        """
        # Define Schema.org namespace
        SCHEMA = Namespace("http://schema.org/")
        SMO = Namespace("http://semiomeme.org/ontology/")

        if not label:
            label = entity_uri.replace('_', ' ')

        # Create the entity node
        entity_node = URIRef(namespace[entity_uri])

        # Add label if not already present
        if (entity_node, RDFS.label, None) not in graph:
            graph.add((entity_node, RDFS.label, Literal(label)))

        # Use entity linker to find external matches
        # UPDATED: Convert ontology class to string for legacy entity linker
        entity_type_string = self._extract_type_from_class(entity_type_class)
        link_result = self.entity_linker.link_entity(label, entity_type_string)

        if not link_result:
            return

        # Try to enrich entities based on their ontology class type
        if link_result.get('source') == 'wikidata':
            # Add owl:sameAs link to Wikidata
            wikidata_uri = link_result.get('wikidata_uri')
            if wikidata_uri:
                graph.add((entity_node, OWL.sameAs, URIRef(wikidata_uri)))

            # UPDATED: Add type-specific enrichment based on ontology class
            self._add_enrichment_by_class(graph, entity_node, entity_type_class, label, link_result)

    def _extract_type_from_class(self, entity_type_class):
        """
        Extract a string type name from an ontology class URI for legacy compatibility.

        Args:
            entity_type_class: URIRef like SMO.PersonEntry

        Returns:
            String like "Person" for legacy entity linker
        """
        if not entity_type_class:
            return "Entity"

        # Extract the class name from the URI
        class_name = str(entity_type_class).split('/')[-1]

        # Map ontology classes to legacy strings
        class_mapping = {
            'PersonEntry': 'Person',
            'MemeEntry': 'Meme',
            'EventEntry': 'Event',
            'SiteEntry': 'Site',
            'SubcultureEntry': 'Subculture',
            'Entity': 'Entity'
        }

        return class_mapping.get(class_name, 'Entity')

    def _add_enrichment_by_class(self, graph, entity_node, entity_type_class, label, link_result):
        """
        Add enrichment properties based on the specific ontology class type.

        Args:
            graph: RDF graphs
            entity_node: The entity being enriched
            entity_type_class: URIRef to ontology class
            label: Entity label
            link_result: Result from entity linking
        """
        SCHEMA = Namespace("http://schema.org/")
        SMO = Namespace("http://semiomeme.org/ontology/")

        # Extract class name for comparison
        class_name = str(entity_type_class).split('/')[-1]

        # Add schema:name for all entities with Wikidata links
        graph.add((entity_node, SCHEMA.name, Literal(label)))

        # Type-specific enrichment
        if class_name == 'PersonEntry':
            self._enrich_person_entry(graph, entity_node, label, link_result)
        elif class_name == 'EventEntry':
            self._enrich_event_entry(graph, entity_node, label, link_result)
        elif class_name == 'SiteEntry':
            self._enrich_site_entry(graph, entity_node, label, link_result)
        elif class_name == 'MemeEntry':
            self._enrich_meme_entry(graph, entity_node, label, link_result)
        elif class_name == 'SubcultureEntry':
            self._enrich_subculture_entry(graph, entity_node, label, link_result)
        elif class_name == 'Entity':
            self._enrich_entity_entry(graph, entity_node, label, link_result)


    def _enrich_person_entry(self, graph, entity_node, label, link_result):
        """Add person-specific enrichment properties."""
        SCHEMA = Namespace("http://schema.org/")
        FOAF = Namespace("http://xmlns.com/foaf/0.1/")

        # Add FOAF name property for people
        graph.add((entity_node, FOAF.name, Literal(label)))

        # Could add more person-specific properties here based on Wikidata results
        # e.g., birthDate, occupation, etc. if available in link_result

    def _enrich_event_entry(self, graph, entity_node, label, link_result):
        """Add event-specific enrichment properties."""
        SCHEMA = Namespace("http://schema.org/")

        # Add event name
        graph.add((entity_node, SCHEMA.eventName, Literal(label)))

    def _enrich_site_entry(self, graph, entity_node, label, link_result):
        """Add site/website-specific enrichment properties."""
        SCHEMA = Namespace("http://schema.org/")

        # Add website name
        graph.add((entity_node, SCHEMA.alternateName, Literal(label)))

    def _enrich_meme_entry(self, graph, entity_node, label, link_result):
        """Add meme-specific enrichment properties."""
        SCHEMA = Namespace("http://schema.org/")
        DCTERMS = Namespace("http://purl.org/dc/terms/")

        # Memes are creative works
        graph.add((entity_node, RDF.type, SCHEMA.CreativeWork))

        # Add title as proper schema property
        graph.add((entity_node, SCHEMA.headline, Literal(label)))

        # If we have Wikidata info, might have creation date
        if link_result and 'inception' in link_result:
            graph.add((entity_node, DCTERMS.created, Literal(link_result['inception'])))

    def _enrich_subculture_entry(self, graph, entity_node, label, link_result):
        """Add subculture-specific enrichment properties."""
        SCHEMA = Namespace("http://schema.org/")
        FOAF = Namespace("http://xmlns.com/foaf/0.1/")

        # Subcultures are communities/groups
        graph.add((entity_node, RDF.type, FOAF.Group))
        graph.add((entity_node, RDF.type, SCHEMA.Organization))

        # Add community name
        graph.add((entity_node, FOAF.name, Literal(label)))
        graph.add((entity_node, SCHEMA.name, Literal(label)))

        # If it's an online community
        if any(term in label.lower() for term in ['reddit', 'chan', 'board', 'forum']):
            graph.add((entity_node, RDF.type, SCHEMA.OnlineCommunity))

    def _enrich_entity_entry(self, graph, entity_node, label, link_result):
        """Add generic entity enrichment properties."""
        SCHEMA = Namespace("http://schema.org/")
        SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

        # Generic entities get basic semantic annotation
        graph.add((entity_node, SKOS.prefLabel, Literal(label)))

        # Mark as a Thing (most generic Schema.org type)
        graph.add((entity_node, RDF.type, SCHEMA.Thing))

        # If Wikidata provided a description
        if link_result and 'description' in link_result:
            graph.add((entity_node, SCHEMA.description, Literal(link_result['description'])))