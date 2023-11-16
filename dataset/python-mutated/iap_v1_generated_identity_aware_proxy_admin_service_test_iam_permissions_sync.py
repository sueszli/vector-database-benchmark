from google.cloud import iap_v1
from google.iam.v1 import iam_policy_pb2

def sample_test_iam_permissions():
    if False:
        return 10
    client = iap_v1.IdentityAwareProxyAdminServiceClient()
    request = iam_policy_pb2.TestIamPermissionsRequest(resource='resource_value', permissions=['permissions_value1', 'permissions_value2'])
    response = client.test_iam_permissions(request=request)
    print(response)