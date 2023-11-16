from google.cloud import securesourcemanager_v1
from google.iam.v1 import iam_policy_pb2

def sample_set_iam_policy_repo():
    if False:
        for i in range(10):
            print('nop')
    client = securesourcemanager_v1.SecureSourceManagerClient()
    request = iam_policy_pb2.SetIamPolicyRequest(resource='resource_value')
    response = client.set_iam_policy_repo(request=request)
    print(response)