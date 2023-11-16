from google.cloud import bigquery_datapolicies_v1

def sample_get_data_policy():
    if False:
        while True:
            i = 10
    client = bigquery_datapolicies_v1.DataPolicyServiceClient()
    request = bigquery_datapolicies_v1.GetDataPolicyRequest(name='name_value')
    response = client.get_data_policy(request=request)
    print(response)