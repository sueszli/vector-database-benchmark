from google.cloud import datacatalog_v1
from google.iam.v1 import iam_policy_pb2

def sample_test_iam_permissions():
    if False:
        while True:
            i = 10
    client = datacatalog_v1.PolicyTagManagerClient()
    request = iam_policy_pb2.TestIamPermissionsRequest(resource='resource_value', permissions=['permissions_value1', 'permissions_value2'])
    response = client.test_iam_permissions(request=request)
    print(response)