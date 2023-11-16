from google.cloud import bigquery_datapolicies_v1

def sample_create_data_policy():
    if False:
        i = 10
        return i + 15
    client = bigquery_datapolicies_v1.DataPolicyServiceClient()
    data_policy = bigquery_datapolicies_v1.DataPolicy()
    data_policy.policy_tag = 'policy_tag_value'
    data_policy.data_masking_policy.predefined_expression = 'DATE_YEAR_MASK'
    request = bigquery_datapolicies_v1.CreateDataPolicyRequest(parent='parent_value', data_policy=data_policy)
    response = client.create_data_policy(request=request)
    print(response)