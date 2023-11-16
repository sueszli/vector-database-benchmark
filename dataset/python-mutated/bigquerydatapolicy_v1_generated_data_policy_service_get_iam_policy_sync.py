from google.cloud import bigquery_datapolicies_v1
from google.iam.v1 import iam_policy_pb2

def sample_get_iam_policy():
    if False:
        while True:
            i = 10
    client = bigquery_datapolicies_v1.DataPolicyServiceClient()
    request = iam_policy_pb2.GetIamPolicyRequest(resource='resource_value')
    response = client.get_iam_policy(request=request)
    print(response)