from google.cloud.devtools import containeranalysis_v1
from grafeas.grafeas_v1 import types

def get_occurrence(occurrence_id: str, project_id: str) -> types.grafeas.Occurrence:
    if False:
        i = 10
        return i + 15
    'retrieves and prints a specified occurrence from the server.'
    client = containeranalysis_v1.ContainerAnalysisClient()
    grafeas_client = client.get_grafeas_client()
    parent = f'projects/{project_id}/occurrences/{occurrence_id}'
    return grafeas_client.get_occurrence(name=parent)