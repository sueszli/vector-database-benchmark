from google.cloud import run_v2
from google.iam.v1 import iam_policy_pb2

def sample_get_iam_policy():
    if False:
        for i in range(10):
            print('nop')
    client = run_v2.ServicesClient()
    request = iam_policy_pb2.GetIamPolicyRequest(resource='resource_value')
    response = client.get_iam_policy(request=request)
    print(response)