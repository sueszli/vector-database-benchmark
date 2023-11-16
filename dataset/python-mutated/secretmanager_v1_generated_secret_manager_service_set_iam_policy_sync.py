from google.cloud import secretmanager_v1
from google.iam.v1 import iam_policy_pb2

def sample_set_iam_policy():
    if False:
        while True:
            i = 10
    client = secretmanager_v1.SecretManagerServiceClient()
    request = iam_policy_pb2.SetIamPolicyRequest(resource='resource_value')
    response = client.set_iam_policy(request=request)
    print(response)