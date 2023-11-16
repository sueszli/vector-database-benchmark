from typing import Dict, List

def list_datasets(project_id: str, location: str) -> List[Dict[str, str]]:
    if False:
        print('Hello World!')
    'Lists the datasets in the project.\n\n    See\n    https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/datasets\n    before running the sample.\n    See https://googleapis.github.io/google-api-python-client/docs/dyn/healthcare_v1.projects.locations.datasets.html#list\n    for the Python API reference.\n\n    Args:\n      project_id: The project ID or project number of the Google Cloud project you want\n          to use.\n      location: The name of the location where the datasets are located.\n\n    Returns:\n      A list of Dataset resources.\n    '
    from googleapiclient import discovery
    from googleapiclient.errors import HttpError
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    dataset_parent = f'projects/{project_id}/locations/{location}'
    datasets = []
    request = client.projects().locations().datasets().list(parent=dataset_parent)
    while request is not None:
        try:
            response = request.execute()
            if response and 'datasets' in response:
                datasets.extend(response['datasets'])
            request = client.projects().locations().datasets().list_next(previous_request=request, previous_response=response)
            for dataset in datasets:
                print(f"Dataset: {dataset.get('name')}\nTime zone: {dataset.get('timeZone')}")
            return datasets
        except HttpError as err:
            raise err