from google.cloud import resourcemanager_v3
from google.iam.v1 import iam_policy_pb2

def sample_get_iam_policy():
    if False:
        i = 10
        return i + 15
    client = resourcemanager_v3.TagKeysClient()
    request = iam_policy_pb2.GetIamPolicyRequest(resource='resource_value')
    response = client.get_iam_policy(request=request)
    print(response)