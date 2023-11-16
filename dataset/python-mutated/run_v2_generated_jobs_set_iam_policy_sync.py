from google.cloud import run_v2
from google.iam.v1 import iam_policy_pb2

def sample_set_iam_policy():
    if False:
        print('Hello World!')
    client = run_v2.JobsClient()
    request = iam_policy_pb2.SetIamPolicyRequest(resource='resource_value')
    response = client.set_iam_policy(request=request)
    print(response)