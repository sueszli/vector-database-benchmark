from google.cloud import tasks_v2beta3
from google.iam.v1 import iam_policy_pb2

def sample_get_iam_policy():
    if False:
        while True:
            i = 10
    client = tasks_v2beta3.CloudTasksClient()
    request = iam_policy_pb2.GetIamPolicyRequest(resource='resource_value')
    response = client.get_iam_policy(request=request)
    print(response)