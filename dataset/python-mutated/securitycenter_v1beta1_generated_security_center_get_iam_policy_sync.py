from google.cloud import securitycenter_v1beta1
from google.iam.v1 import iam_policy_pb2

def sample_get_iam_policy():
    if False:
        for i in range(10):
            print('nop')
    client = securitycenter_v1beta1.SecurityCenterClient()
    request = iam_policy_pb2.GetIamPolicyRequest(resource='resource_value')
    response = client.get_iam_policy(request=request)
    print(response)