def delete_dataset(project_id: str, location: str, dataset_id: str) -> None:
    if False:
        while True:
            i = 10
    "Deletes a dataset.\n\n    See\n    https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/datasets\n    before running the sample.\n    See https://googleapis.github.io/google-api-python-client/docs/dyn/healthcare_v1.projects.locations.datasets.html#delete\n    for the Python API reference.\n\n    Args:\n      project_id: The project ID or project number of the Google Cloud project you want\n          to use.\n      location: The name of the dataset's location.\n      dataset_id: The name of the dataset to delete.\n\n    Returns:\n      An empty response body.\n    "
    from googleapiclient import discovery
    from googleapiclient.errors import HttpError
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    dataset_name = f'projects/{project_id}/locations/{location}/datasets/{dataset_id}'
    request = client.projects().locations().datasets().delete(name=dataset_name)
    try:
        request.execute()
        print(f'Deleted dataset: {dataset_id}')
    except HttpError as err:
        raise err