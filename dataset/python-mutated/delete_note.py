from google.cloud.devtools import containeranalysis_v1

def delete_note(note_id: str, project_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Removes an existing note from the server.'
    client = containeranalysis_v1.ContainerAnalysisClient()
    grafeas_client = client.get_grafeas_client()
    note_name = f'projects/{project_id}/notes/{note_id}'
    grafeas_client.delete_note(name=note_name)