from typing import List
from grafeas.grafeas_v1 import types

def find_high_severity_vulnerabilities_for_image(resource_url: str, project_id: str) -> List[types.grafeas.Occurrence]:
    if False:
        print('Hello World!')
    'Retrieves a list of only high vulnerability occurrences associated\n    with a resource.'
    from grafeas.grafeas_v1 import Severity
    from google.cloud.devtools import containeranalysis_v1
    client = containeranalysis_v1.ContainerAnalysisClient()
    grafeas_client = client.get_grafeas_client()
    project_name = f'projects/{project_id}'
    filter_str = 'kind="VULNERABILITY" AND resourceUrl="{}"'.format(resource_url)
    vulnerabilities = grafeas_client.list_occurrences(parent=project_name, filter=filter_str)
    filtered_list = []
    for v in vulnerabilities:
        if v.vulnerability.effective_severity == Severity.HIGH or v.vulnerability.effective_severity == Severity.CRITICAL:
            filtered_list.append(v)
    return filtered_list