from google.cloud import artifactregistry_v1beta2
from google.iam.v1 import iam_policy_pb2

def sample_get_iam_policy():
    if False:
        for i in range(10):
            print('nop')
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = iam_policy_pb2.GetIamPolicyRequest(resource='resource_value')
    response = client.get_iam_policy(request=request)
    print(response)