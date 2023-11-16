from typing import List
from google.cloud.devtools import containeranalysis_v1
from grafeas.grafeas_v1 import types

def find_vulnerabilities_for_image(resource_url: str, project_id: str) -> List[types.grafeas.Occurrence]:
    if False:
        i = 10
        return i + 15
    ' "Retrieves all vulnerability occurrences associated with a resource.'
    client = containeranalysis_v1.ContainerAnalysisClient()
    grafeas_client = client.get_grafeas_client()
    project_name = f'projects/{project_id}'
    filter_str = 'kind="VULNERABILITY" AND resourceUrl="{}"'.format(resource_url)
    return list(grafeas_client.list_occurrences(parent=project_name, filter=filter_str))