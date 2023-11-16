import json
from typing import Dict, List, Optional, Tuple
from django.utils import timezone
from posthog.models import EventDefinition, EventProperty, PropertyDefinition
from posthog.models.group.sql import GROUPS_TABLE
from posthog.models.person.sql import PERSONS_TABLE
from posthog.models.property_definition import PropertyType

def infer_taxonomy_for_team(team_id: int) -> Tuple[int, int, int]:
    if False:
        for i in range(10):
            print('nop')
    'Infer event and property definitions based on ClickHouse data.\n\n    In production, the plugin server is responsible for this - but in demo data we insert directly to ClickHouse.\n    '
    events_last_seen_at = _get_events_last_seen_at(team_id)
    event_definitions = EventDefinition.objects.bulk_create([EventDefinition(team_id=team_id, name=event, last_seen_at=last_seen_at) for (event, last_seen_at) in events_last_seen_at.items()], batch_size=1000, ignore_conflicts=True)
    property_types = _get_property_types(team_id)
    property_definitions = PropertyDefinition.objects.bulk_create([PropertyDefinition(team_id=team_id, name=property_key, property_type=property_type, is_numerical=property_type == PropertyType.Numeric) for (property_key, property_type) in property_types.items()], batch_size=1000, ignore_conflicts=True)
    event_property_pairs = _get_event_property_pairs(team_id)
    event_properties = EventProperty.objects.bulk_create([EventProperty(team_id=team_id, event=event, property=property_key) for (event, property_key) in event_property_pairs], batch_size=1000, ignore_conflicts=True)
    return (len(event_definitions), len(property_definitions), len(event_properties))

def _get_events_last_seen_at(team_id: int) -> Dict[str, timezone.datetime]:
    if False:
        print('Hello World!')
    from posthog.client import sync_execute
    return {event: last_seen_at for (event, last_seen_at) in sync_execute(_GET_EVENTS_LAST_SEEN_AT, {'team_id': team_id})}

def _get_property_types(team_id: int) -> Dict[str, Optional[PropertyType]]:
    if False:
        print('Hello World!')
    'Determine property types based on ClickHouse data.'
    from posthog.client import sync_execute
    property_types = {property_key: _infer_property_type(sample_json_value) for (property_key, sample_json_value) in sync_execute(_GET_EVENT_PROPERTY_SAMPLE_JSON_VALUES, {'team_id': team_id})}
    for (property_key, sample_json_value) in sync_execute(_GET_PERSON_PROPERTY_SAMPLE_JSON_VALUES, {'team_id': team_id}):
        if property_key not in property_types:
            property_types[property_key] = _infer_property_type(sample_json_value)
    for (property_key, sample_json_value) in sync_execute(_GET_GROUP_PROPERTY_SAMPLE_JSON_VALUES, {'team_id': team_id}):
        if property_key not in property_types:
            property_types[property_key] = _infer_property_type(sample_json_value)
    return property_types

def _infer_property_type(sample_json_value: str) -> Optional[PropertyType]:
    if False:
        while True:
            i = 10
    'Parse the provided sample value as JSON and return its property type.'
    parsed_value = json.loads(sample_json_value)
    if isinstance(parsed_value, bool):
        return PropertyType.Boolean
    if isinstance(parsed_value, (float, int)):
        return PropertyType.Numeric
    if isinstance(parsed_value, str):
        return PropertyType.String
    return None

def _get_event_property_pairs(team_id: int) -> List[Tuple[str, str]]:
    if False:
        print('Hello World!')
    'Determine which properties have been since with which events based on ClickHouse data.'
    from posthog.client import sync_execute
    return [row[0] for row in sync_execute(_GET_EVENT_PROPERTIES, {'team_id': team_id})]
_GET_EVENTS_LAST_SEEN_AT = '\nSELECT event, max(timestamp) AS last_seen_at\nFROM events\nWHERE team_id = %(team_id)s\nGROUP BY event\n'
_GET_EVENT_PROPERTY_SAMPLE_JSON_VALUES = '\nWITH property_tuples AS (\n    SELECT DISTINCT ON (property_tuple.1) arrayJoin(JSONExtractKeysAndValuesRaw(properties)) AS property_tuple\n    FROM events\n    WHERE team_id = %(team_id)s\n)\nSELECT property_tuple.1 AS property_key, property_tuple.2 AS sample_json_value FROM property_tuples\n'
_GET_ACTOR_PROPERTY_SAMPLE_JSON_VALUES = '\nWITH property_tuples AS (\n    SELECT arrayJoin(JSONExtractKeysAndValuesRaw({properties_column})) AS property_key_value_pair FROM {table_name}\n    WHERE team_id = %(team_id)s\n)\nSELECT\n    property_key_value_pair.1 AS property_key,\n    anyLast(property_key_value_pair.2) AS sample_json_value\nFROM property_tuples\nGROUP BY property_key\n'
_GET_PERSON_PROPERTY_SAMPLE_JSON_VALUES = _GET_ACTOR_PROPERTY_SAMPLE_JSON_VALUES.format(table_name=PERSONS_TABLE, properties_column='properties')
_GET_GROUP_PROPERTY_SAMPLE_JSON_VALUES = _GET_ACTOR_PROPERTY_SAMPLE_JSON_VALUES.format(table_name=GROUPS_TABLE, properties_column='group_properties')
_GET_EVENT_PROPERTIES = '\nSELECT DISTINCT (event, arrayJoin(JSONExtractKeys(properties))) FROM events\nWHERE team_id = %(team_id)s\n'