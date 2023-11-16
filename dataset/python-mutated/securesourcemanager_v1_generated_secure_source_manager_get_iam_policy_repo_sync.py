from google.cloud import securesourcemanager_v1
from google.iam.v1 import iam_policy_pb2

def sample_get_iam_policy_repo():
    if False:
        while True:
            i = 10
    client = securesourcemanager_v1.SecureSourceManagerClient()
    request = iam_policy_pb2.GetIamPolicyRequest(resource='resource_value')
    response = client.get_iam_policy_repo(request=request)
    print(response)