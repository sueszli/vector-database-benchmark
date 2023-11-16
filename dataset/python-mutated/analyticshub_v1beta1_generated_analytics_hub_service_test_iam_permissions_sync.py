from google.cloud import bigquery_data_exchange_v1beta1
from google.iam.v1 import iam_policy_pb2

def sample_test_iam_permissions():
    if False:
        while True:
            i = 10
    client = bigquery_data_exchange_v1beta1.AnalyticsHubServiceClient()
    request = iam_policy_pb2.TestIamPermissionsRequest(resource='resource_value', permissions=['permissions_value1', 'permissions_value2'])
    response = client.test_iam_permissions(request=request)
    print(response)