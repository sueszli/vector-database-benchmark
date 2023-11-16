import time
from google.cloud.devtools import containeranalysis_v1
from grafeas.grafeas_v1 import DiscoveryOccurrence

def poll_discovery_finished(resource_url: str, timeout_seconds: int, project_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Returns the discovery occurrence for a resource once it reaches a\n    terminal state.'
    deadline = time.time() + timeout_seconds
    client = containeranalysis_v1.ContainerAnalysisClient()
    grafeas_client = client.get_grafeas_client()
    project_name = f'projects/{project_id}'
    discovery_occurrence = None
    while discovery_occurrence is None:
        time.sleep(1)
        filter_str = 'resourceUrl="{}"                       AND noteProjectId="goog-analysis"                       AND noteId="PACKAGE_VULNERABILITY"'.format(resource_url)
        filter_str = 'kind="DISCOVERY" AND resourceUrl="{}"'.format(resource_url)
        result = grafeas_client.list_occurrences(parent=project_name, filter=filter_str)
        for item in result:
            discovery_occurrence = item
        if time.time() > deadline:
            raise RuntimeError('timeout while retrieving discovery occurrence')
    status = DiscoveryOccurrence.AnalysisStatus.PENDING
    while status != DiscoveryOccurrence.AnalysisStatus.FINISHED_UNSUPPORTED and status != DiscoveryOccurrence.AnalysisStatus.FINISHED_FAILED and (status != DiscoveryOccurrence.AnalysisStatus.FINISHED_SUCCESS):
        time.sleep(1)
        updated = grafeas_client.get_occurrence(name=discovery_occurrence.name)
        status = updated.discovery.analysis_status
        if time.time() > deadline:
            raise RuntimeError('timeout while waiting for terminal state')
    return discovery_occurrence