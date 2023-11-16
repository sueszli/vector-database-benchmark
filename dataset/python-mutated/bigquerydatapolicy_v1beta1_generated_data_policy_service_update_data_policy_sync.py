from google.cloud import bigquery_datapolicies_v1beta1

def sample_update_data_policy():
    if False:
        print('Hello World!')
    client = bigquery_datapolicies_v1beta1.DataPolicyServiceClient()
    data_policy = bigquery_datapolicies_v1beta1.DataPolicy()
    data_policy.policy_tag = 'policy_tag_value'
    data_policy.data_masking_policy.predefined_expression = 'DEFAULT_MASKING_VALUE'
    request = bigquery_datapolicies_v1beta1.UpdateDataPolicyRequest(data_policy=data_policy)
    response = client.update_data_policy(request=request)
    print(response)