from google.cloud import securesourcemanager_v1
from google.iam.v1 import iam_policy_pb2

def sample_test_iam_permissions_repo():
    if False:
        for i in range(10):
            print('nop')
    client = securesourcemanager_v1.SecureSourceManagerClient()
    request = iam_policy_pb2.TestIamPermissionsRequest(resource='resource_value', permissions=['permissions_value1', 'permissions_value2'])
    response = client.test_iam_permissions_repo(request=request)
    print(response)