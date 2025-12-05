from config.namespaces import SMO

class EntityTypeResolver:
    """
    Simple entity type resolver for hierarchical ontology approach.
    Focuses on preserving primary types from KYM data.
    """

    def __init__(self):
        # Define type hierarchy - higher index = higher priority when conflicts occur

        self.type_hierarchy = [
            SMO.Entity, SMO.Tag, SMO.Series,
            SMO.MemeType,  # ADD SMO.MemeFormat here
            SMO.EntryType, SMO.Status, SMO.Badge,
            SMO.SiteEntry, SMO.SubcultureEntry, SMO.EventEntry,
            SMO.PersonEntry, SMO.MemeEntry
        ]

    def resolve_entity_type_conflict(self, entity_uri, new_type, entity_registry, related_fields=None):
        """
        Resolve entity type conflicts using type hierarchies and context.

        Args:
            entity_uri: The URI string (without namespace)
            new_type: The proposed new type
            entity_registry: Registry of known entities
            related_fields: Additional context about the entity

        Returns:
            The resolved entity type to use
        """
        # If entity doesn't exist yet, use the new type
        if entity_uri not in entity_registry:
            return new_type

        existing_type = entity_registry[entity_uri]['type']

        # If types are the same, no conflict
        if existing_type == new_type:
            return existing_type

        # Use contextual analysis if available
        if related_fields:
            resolved_type = self._resolve_with_context(entity_uri, existing_type, new_type, related_fields)
            if resolved_type:
                return resolved_type

        # Fall back to type hierarchy
        return self._resolve_with_hierarchy(existing_type, new_type)

    def _resolve_with_context(self, entity_uri, existing_type, new_type, related_fields):
        """Use context to resolve type conflicts."""

        # Check relation context
        if 'relation' in related_fields:
            relation = related_fields['relation'].lower()

            # Person-related relations
            if relation in ['creator', 'author', 'performer', 'developer']:
                if 'Person' in [existing_type, new_type]:
                    return 'Person'

            # Type-related relations
            if relation in ['has_type']:
                if 'MemeType' in [existing_type, new_type]:
                    return 'MemeType'

            # Tag-related relations
            if relation in ['has_tag']:
                if 'Tag' in [existing_type, new_type]:
                    return 'Tag'

        # Check source type context
        if 'source_type' in related_fields:
            source_type = related_fields['source_type']

            # If source is a Person and we're considering Person type
            if source_type == 'Person' and 'Person' in [existing_type, new_type]:
                return 'Person'

        return None  # No context-based resolution

    def _resolve_with_hierarchy(self, existing_type, new_type):
        """Resolve conflict using type hierarchy."""
        try:
            existing_priority = self.type_hierarchy.index(existing_type)
        except ValueError:
            existing_priority = -1

        try:
            new_priority = self.type_hierarchy.index(new_type)
        except ValueError:
            new_priority = -1

        # Use the higher priority type (higher index = higher priority)
        if new_priority > existing_priority:
            return new_type
        else:
            return existing_type