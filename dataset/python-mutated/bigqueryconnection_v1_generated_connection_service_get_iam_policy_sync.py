from google.cloud import bigquery_connection_v1
from google.iam.v1 import iam_policy_pb2

def sample_get_iam_policy():
    if False:
        i = 10
        return i + 15
    client = bigquery_connection_v1.ConnectionServiceClient()
    request = iam_policy_pb2.GetIamPolicyRequest(resource='resource_value')
    response = client.get_iam_policy(request=request)
    print(response)