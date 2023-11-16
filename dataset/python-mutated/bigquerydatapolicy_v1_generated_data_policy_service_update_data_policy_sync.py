from google.cloud import bigquery_datapolicies_v1

def sample_update_data_policy():
    if False:
        i = 10
        return i + 15
    client = bigquery_datapolicies_v1.DataPolicyServiceClient()
    data_policy = bigquery_datapolicies_v1.DataPolicy()
    data_policy.policy_tag = 'policy_tag_value'
    data_policy.data_masking_policy.predefined_expression = 'DATE_YEAR_MASK'
    request = bigquery_datapolicies_v1.UpdateDataPolicyRequest(data_policy=data_policy)
    response = client.update_data_policy(request=request)
    print(response)