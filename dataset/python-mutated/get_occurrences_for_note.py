from google.cloud.devtools import containeranalysis_v1

def get_occurrences_for_note(note_id: str, project_id: str) -> int:
    if False:
        i = 10
        return i + 15
    'Retrieves all the occurrences associated with a specified Note.\n    Here, all occurrences are printed and counted.'
    client = containeranalysis_v1.ContainerAnalysisClient()
    grafeas_client = client.get_grafeas_client()
    note_name = f'projects/{project_id}/notes/{note_id}'
    response = grafeas_client.list_note_occurrences(name=note_name)
    count = 0
    for o in response:
        count += 1
    return count