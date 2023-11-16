from typing import Dict

def create_dataset(project_id: str, location: str, dataset_id: str) -> Dict[str, str]:
    if False:
        print('Hello World!')
    "Creates a Cloud Healthcare API dataset.\n\n    See\n    https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/datasets\n    before running the sample.\n    See\n    https://googleapis.github.io/google-api-python-client/docs/dyn/healthcare_v1.projects.locations.datasets.html#create\n    for the Python API reference.\n\n    Args:\n      project_id: The project ID or project number of the Google Cloud project you want\n          to use.\n      location: The name of the dataset's location.\n      dataset_id: The ID of the dataset to create.\n\n    Returns:\n      A dictionary representing a long-running operation that results from\n      calling the 'CreateDataset' method. Dataset creation is typically fast.\n    "
    import time
    from googleapiclient import discovery
    from googleapiclient.errors import HttpError
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    dataset_parent = f'projects/{project_id}/locations/{location}'
    request = client.projects().locations().datasets().create(parent=dataset_parent, body={}, datasetId=dataset_id)
    start_time = time.time()
    max_time = 600
    try:
        operation = request.execute()
        while not operation.get('done', False):
            print('Waiting for operation to finish...')
            if time.time() - start_time > max_time:
                raise TimeoutError('Timed out waiting for operation to finish.')
            operation = client.projects().locations().datasets().operations().get(name=operation['name']).execute()
            time.sleep(5)
        if 'error' in operation:
            raise RuntimeError(f"Create dataset operation failed: {operation['error']}")
        else:
            dataset_name = operation['response']['name']
            print(f'Created dataset: {dataset_name}')
            return operation
    except HttpError as err:
        if err.resp.status == 409:
            print(f'Dataset with ID {dataset_id} already exists.')
            return
        else:
            raise err