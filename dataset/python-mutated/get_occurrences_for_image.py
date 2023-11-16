from google.cloud.devtools import containeranalysis_v1

def get_occurrences_for_image(resource_url: str, project_id: str) -> int:
    if False:
        while True:
            i = 10
    'Retrieves all the occurrences associated with a specified image.\n    Here, all occurrences are simply printed and counted.'
    filter_str = f'resourceUrl="{resource_url}"'
    client = containeranalysis_v1.ContainerAnalysisClient()
    grafeas_client = client.get_grafeas_client()
    project_name = f'projects/{project_id}'
    response = grafeas_client.list_occurrences(parent=project_name, filter=filter_str)
    count = 0
    for o in response:
        count += 1
    return count