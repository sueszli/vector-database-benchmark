from google.cloud import functions_v1
from google.iam.v1 import iam_policy_pb2

def sample_set_iam_policy():
    if False:
        for i in range(10):
            print('nop')
    client = functions_v1.CloudFunctionsServiceClient()
    request = iam_policy_pb2.SetIamPolicyRequest(resource='resource_value')
    response = client.set_iam_policy(request=request)
    print(response)