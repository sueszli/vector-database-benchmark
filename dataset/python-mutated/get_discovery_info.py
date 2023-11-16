from google.cloud.devtools import containeranalysis_v1

def get_discovery_info(resource_url: str, project_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Retrieves and prints the discovery occurrence created for a specified\n    image. The discovery occurrence contains information about the initial\n    scan on the image.'
    filter_str = f'kind="DISCOVERY" AND resourceUrl="{resource_url}"'
    client = containeranalysis_v1.ContainerAnalysisClient()
    grafeas_client = client.get_grafeas_client()
    project_name = f'projects/{project_id}'
    response = grafeas_client.list_occurrences(parent=project_name, filter_=filter_str)
    for occ in response:
        print(occ)