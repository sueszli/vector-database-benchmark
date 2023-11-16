from google.cloud import bigquery_datapolicies_v1

def sample_rename_data_policy():
    if False:
        return 10
    client = bigquery_datapolicies_v1.DataPolicyServiceClient()
    request = bigquery_datapolicies_v1.RenameDataPolicyRequest(name='name_value', new_data_policy_id='new_data_policy_id_value')
    response = client.rename_data_policy(request=request)
    print(response)