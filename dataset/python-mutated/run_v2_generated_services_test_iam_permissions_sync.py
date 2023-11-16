from google.cloud import run_v2
from google.iam.v1 import iam_policy_pb2

def sample_test_iam_permissions():
    if False:
        while True:
            i = 10
    client = run_v2.ServicesClient()
    request = iam_policy_pb2.TestIamPermissionsRequest(resource='resource_value', permissions=['permissions_value1', 'permissions_value2'])
    response = client.test_iam_permissions(request=request)
    print(response)