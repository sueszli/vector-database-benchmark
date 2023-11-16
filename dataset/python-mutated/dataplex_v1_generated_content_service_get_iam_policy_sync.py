from google.cloud import dataplex_v1
from google.iam.v1 import iam_policy_pb2

def sample_get_iam_policy():
    if False:
        return 10
    client = dataplex_v1.ContentServiceClient()
    request = iam_policy_pb2.GetIamPolicyRequest(resource='resource_value')
    response = client.get_iam_policy(request=request)
    print(response)