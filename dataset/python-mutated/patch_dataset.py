from typing import Dict

def patch_dataset(project_id: str, location: str, dataset_id: str, time_zone: str) -> Dict[str, str]:
    if False:
        i = 10
        return i + 15
    "Updates dataset metadata.\n\n    See\n    https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/datasets\n    before running the sample.\n    See https://googleapis.github.io/google-api-python-client/docs/dyn/healthcare_v1.projects.locations.datasets.html#patch\n    for the Python API reference.\n\n    Args:\n      project_id: The project ID or project number of the Google Cloud project you want\n          to use.\n      location: The name of the dataset's location.\n      dataset_id: The ID of the dataset to patch.\n      time_zone: The default timezone used by the dataset.\n\n    Returns:\n      A dictionary representing the patched Dataset resource.\n    "
    from googleapiclient import discovery
    from googleapiclient.errors import HttpError
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    dataset_parent = f'projects/{project_id}/locations/{location}'
    dataset_name = f'{dataset_parent}/datasets/{dataset_id}'
    patch = {'timeZone': time_zone}
    request = client.projects().locations().datasets().patch(name=dataset_name, updateMask='timeZone', body=patch)
    try:
        response = request.execute()
        print(f'Patched dataset {dataset_id} with time zone: {time_zone}')
        return response
    except HttpError as err:
        raise err