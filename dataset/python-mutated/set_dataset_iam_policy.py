from typing import Any, Dict

def set_dataset_iam_policy(project_id: str, location: str, dataset_id: str, member: str, role: str, etag: str=None) -> Dict[str, Any]:
    if False:
        while True:
            i = 10
    "Sets the IAM policy for the specified dataset.\n\n        A single member will be assigned a single role. A member can be any of:\n\n        - allUsers, that is, anyone\n        - allAuthenticatedUsers, anyone authenticated with a Google account\n        - user:email, as in 'user:somebody@example.com'\n        - group:email, as in 'group:admins@example.com'\n        - domain:domainname, as in 'domain:example.com'\n        - serviceAccount:email,\n            as in 'serviceAccount:my-other-app@appspot.gserviceaccount.com'\n\n        A role can be any IAM role, such as 'roles/viewer', 'roles/owner',\n        or 'roles/editor'\n\n    See\n    https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/datasets\n    before running the sample.\n    See https://googleapis.github.io/google-api-python-client/docs/dyn/healthcare_v1.projects.locations.datasets.html#setIamPolicy\n    for the Python API reference.\n\n    Args:\n      project_id: The project ID or project number of the Google Cloud project you want\n          to use.\n      location: The name of the dataset's location.\n      dataset_id: The ID of the dataset containing the IAM policy to set.\n      member: The principals to grant access for a Google Cloud resource.\n      role: The role to assign to the list of 'members'.\n      etag: The 'etag' returned in a previous getIamPolicy request to ensure that\n        setIamPolicy changes apply to the same policy version.\n\n    Returns:\n      A dictionary representing an IAM policy.\n    "
    from googleapiclient import discovery
    from googleapiclient.errors import HttpError
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    dataset_name = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    policy = {'bindings': [{'role': role, 'members': [member]}]}
    if etag is not None:
        policy['etag'] = etag
    request = client.projects().locations().datasets().setIamPolicy(resource=dataset_name, body={'policy': policy})
    try:
        response = request.execute()
        print('etag: {}'.format(response.get('name')))
        print('bindings: {}'.format(response.get('bindings')))
        return response
    except HttpError as err:
        raise err