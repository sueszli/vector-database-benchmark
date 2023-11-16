from google.cloud import bigquery_datapolicies_v1beta1

def sample_get_data_policy():
    if False:
        i = 10
        return i + 15
    client = bigquery_datapolicies_v1beta1.DataPolicyServiceClient()
    request = bigquery_datapolicies_v1beta1.GetDataPolicyRequest(name='name_value')
    response = client.get_data_policy(request=request)
    print(response)