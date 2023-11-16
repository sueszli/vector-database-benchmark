from google.cloud import tasks_v2
from google.iam.v1 import iam_policy_pb2

def sample_test_iam_permissions():
    if False:
        i = 10
        return i + 15
    client = tasks_v2.CloudTasksClient()
    request = iam_policy_pb2.TestIamPermissionsRequest(resource='resource_value', permissions=['permissions_value1', 'permissions_value2'])
    response = client.test_iam_permissions(request=request)
    print(response)