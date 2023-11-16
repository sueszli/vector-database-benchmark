from google.cloud.devtools import containeranalysis_v1
from grafeas.grafeas_v1 import types

def get_note(note_id: str, project_id: str) -> types.grafeas.Note:
    if False:
        for i in range(10):
            print('nop')
    'Retrieves and prints a specified note from the server.'
    client = containeranalysis_v1.ContainerAnalysisClient()
    grafeas_client = client.get_grafeas_client()
    note_name = f'projects/{project_id}/notes/{note_id}'
    response = grafeas_client.get_note(name=note_name)
    return response