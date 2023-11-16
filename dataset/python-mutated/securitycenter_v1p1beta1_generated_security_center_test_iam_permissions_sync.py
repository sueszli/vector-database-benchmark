from google.cloud import securitycenter_v1p1beta1
from google.iam.v1 import iam_policy_pb2

def sample_test_iam_permissions():
    if False:
        for i in range(10):
            print('nop')
    client = securitycenter_v1p1beta1.SecurityCenterClient()
    request = iam_policy_pb2.TestIamPermissionsRequest(resource='resource_value', permissions=['permissions_value1', 'permissions_value2'])
    response = client.test_iam_permissions(request=request)
    print(response)