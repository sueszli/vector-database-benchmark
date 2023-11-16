from google.cloud import tasks_v2beta3
from google.iam.v1 import iam_policy_pb2

def sample_test_iam_permissions():
    if False:
        for i in range(10):
            print('nop')
    client = tasks_v2beta3.CloudTasksClient()
    request = iam_policy_pb2.TestIamPermissionsRequest(resource='resource_value', permissions=['permissions_value1', 'permissions_value2'])
    response = client.test_iam_permissions(request=request)
    print(response)