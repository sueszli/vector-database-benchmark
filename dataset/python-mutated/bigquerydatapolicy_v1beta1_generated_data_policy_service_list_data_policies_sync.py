from google.cloud import bigquery_datapolicies_v1beta1

def sample_list_data_policies():
    if False:
        i = 10
        return i + 15
    client = bigquery_datapolicies_v1beta1.DataPolicyServiceClient()
    request = bigquery_datapolicies_v1beta1.ListDataPoliciesRequest(parent='parent_value')
    page_result = client.list_data_policies(request=request)
    for response in page_result:
        print(response)