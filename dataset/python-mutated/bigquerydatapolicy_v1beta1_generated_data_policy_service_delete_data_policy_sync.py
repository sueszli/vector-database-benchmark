from google.cloud import bigquery_datapolicies_v1beta1

def sample_delete_data_policy():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_datapolicies_v1beta1.DataPolicyServiceClient()
    request = bigquery_datapolicies_v1beta1.DeleteDataPolicyRequest(name='name_value')
    client.delete_data_policy(request=request)