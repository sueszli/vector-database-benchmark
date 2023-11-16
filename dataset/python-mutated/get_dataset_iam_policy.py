from typing import Any, Dict

def get_dataset_iam_policy(project_id: str, location: str, dataset_id: str) -> Dict[str, Any]:
    if False:
        return 10
    "Gets the IAM policy for the specified dataset.\n\n    See\n    https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/datasets\n    before running the sample.\n    See https://googleapis.github.io/google-api-python-client/docs/dyn/healthcare_v1.projects.locations.datasets.html#getIamPolicy\n    for the Python API reference.\n\n    Args:\n      project_id: The project ID or project number of the Google Cloud project you want\n          to use.\n      location: The name of the dataset's location.\n      dataset_id: The ID of the dataset containing the IAM policy to get.\n\n    Returns:\n      A dictionary representing an IAM policy.\n    "
    from googleapiclient import discovery
    from googleapiclient.errors import HttpError
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    dataset_name = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    request = client.projects().locations().datasets().getIamPolicy(resource=dataset_name)
    try:
        response = request.execute()
        print('etag: {}'.format(response.get('name')))
        return response
    except HttpError as err:
        raise err