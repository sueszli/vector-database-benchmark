from google.cloud.devtools import containeranalysis_v1
from grafeas.grafeas_v1 import types, Version

def create_note(note_id: str, project_id: str) -> types.grafeas.Note:
    if False:
        print('Hello World!')
    'Creates and returns a new vulnerability note.'
    client = containeranalysis_v1.ContainerAnalysisClient()
    grafeas_client = client.get_grafeas_client()
    project_name = f'projects/{project_id}'
    note = {'vulnerability': {'details': [{'affected_cpe_uri': 'your-uri-here', 'affected_package': 'your-package-here', 'affected_version_start': {'kind': Version.VersionKind.MINIMUM}, 'fixed_version': {'kind': Version.VersionKind.MAXIMUM}}]}}
    response = grafeas_client.create_note(parent=project_name, note_id=note_id, note=note)
    return response