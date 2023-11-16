from google.cloud import bigquery_datapolicies_v1

def sample_delete_data_policy():
    if False:
        return 10
    client = bigquery_datapolicies_v1.DataPolicyServiceClient()
    request = bigquery_datapolicies_v1.DeleteDataPolicyRequest(name='name_value')
    client.delete_data_policy(request=request)