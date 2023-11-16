from google.cloud import resourcemanager_v3
from google.iam.v1 import iam_policy_pb2

def sample_get_iam_policy():
    if False:
        for i in range(10):
            print('nop')
    client = resourcemanager_v3.ProjectsClient()
    request = iam_policy_pb2.GetIamPolicyRequest(resource='resource_value')
    response = client.get_iam_policy(request=request)
    print(response)