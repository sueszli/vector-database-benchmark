from google.cloud import securitycenter_v1
from google.iam.v1 import iam_policy_pb2

def sample_get_iam_policy():
    if False:
        return 10
    client = securitycenter_v1.SecurityCenterClient()
    request = iam_policy_pb2.GetIamPolicyRequest(resource='resource_value')
    response = client.get_iam_policy(request=request)
    print(response)