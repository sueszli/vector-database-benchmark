from google.cloud.devtools import containeranalysis_v1

def delete_occurrence(occurrence_id: str, project_id: str) -> None:
    if False:
        print('Hello World!')
    'Removes an existing occurrence from the server.'
    client = containeranalysis_v1.ContainerAnalysisClient()
    grafeas_client = client.get_grafeas_client()
    parent = f'projects/{project_id}/occurrences/{occurrence_id}'
    grafeas_client.delete_occurrence(name=parent)