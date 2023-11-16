from google.cloud import secretmanager_v1
from google.iam.v1 import iam_policy_pb2

def sample_get_iam_policy():
    if False:
        print('Hello World!')
    client = secretmanager_v1.SecretManagerServiceClient()
    request = iam_policy_pb2.GetIamPolicyRequest(resource='resource_value')
    response = client.get_iam_policy(request=request)
    print(response)