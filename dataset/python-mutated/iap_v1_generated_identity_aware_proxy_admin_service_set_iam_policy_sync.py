from google.cloud import iap_v1
from google.iam.v1 import iam_policy_pb2

def sample_set_iam_policy():
    if False:
        return 10
    client = iap_v1.IdentityAwareProxyAdminServiceClient()
    request = iam_policy_pb2.SetIamPolicyRequest(resource='resource_value')
    response = client.set_iam_policy(request=request)
    print(response)