from google.cloud import artifactregistry_v1beta2
from google.iam.v1 import iam_policy_pb2

def sample_set_iam_policy():
    if False:
        print('Hello World!')
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = iam_policy_pb2.SetIamPolicyRequest(resource='resource_value')
    response = client.set_iam_policy(request=request)
    print(response)