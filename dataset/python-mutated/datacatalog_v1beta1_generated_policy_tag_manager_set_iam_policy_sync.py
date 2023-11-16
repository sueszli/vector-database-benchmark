from google.cloud import datacatalog_v1beta1
from google.iam.v1 import iam_policy_pb2

def sample_set_iam_policy():
    if False:
        while True:
            i = 10
    client = datacatalog_v1beta1.PolicyTagManagerClient()
    request = iam_policy_pb2.SetIamPolicyRequest(resource='resource_value')
    response = client.set_iam_policy(request=request)
    print(response)