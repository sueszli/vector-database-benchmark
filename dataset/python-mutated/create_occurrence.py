from google.cloud.devtools import containeranalysis_v1
from grafeas.grafeas_v1 import types, Version

def create_occurrence(resource_url: str, note_id: str, occurrence_project: str, note_project: str) -> types.grafeas.Occurrence:
    if False:
        print('Hello World!')
    'Creates and returns a new occurrence of a previously\n    created vulnerability note.'
    client = containeranalysis_v1.ContainerAnalysisClient()
    grafeas_client = client.get_grafeas_client()
    formatted_note = f'projects/{note_project}/notes/{note_id}'
    formatted_project = f'projects/{occurrence_project}'
    occurrence = {'note_name': formatted_note, 'resource_uri': resource_url, 'vulnerability': {'package_issue': [{'affected_cpe_uri': 'your-uri-here', 'affected_package': 'your-package-here', 'affected_version': {'kind': Version.VersionKind.MINIMUM}, 'fixed_version': {'kind': Version.VersionKind.MAXIMUM}}]}}
    return grafeas_client.create_occurrence(parent=formatted_project, occurrence=occurrence)