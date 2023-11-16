from typing import Dict

def get_dataset(project_id: str, location: str, dataset_id: str) -> Dict[str, str]:
    if False:
        for i in range(10):
            print('nop')
    "Gets any metadata associated with a dataset.\n\n    See\n    https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/datasets\n    before running the sample.\n    See https://googleapis.github.io/google-api-python-client/docs/dyn/healthcare_v1.projects.locations.datasets.html#get\n    for the Python API reference.\n\n    Args:\n      project_id: The project ID or project number of the Google Cloud project you want\n          to use.\n      location: The name of the dataset's location.\n      dataset_id: The name of the dataset to get.\n\n    Returns:\n      A dictionary representing a Dataset resource.\n    "
    from googleapiclient import discovery
    from googleapiclient.errors import HttpError
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    dataset_name = f'projects/{project_id}/locations/{location}/datasets/{dataset_id}'
    request = client.projects().locations().datasets()
    try:
        dataset = request.get(name=dataset_name).execute()
        print(f"Name: {dataset.get('name')}")
        return dataset
    except HttpError as err:
        raise err