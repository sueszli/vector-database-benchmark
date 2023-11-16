from google.cloud import bigquery_datapolicies_v1

def sample_list_data_policies():
    if False:
        print('Hello World!')
    client = bigquery_datapolicies_v1.DataPolicyServiceClient()
    request = bigquery_datapolicies_v1.ListDataPoliciesRequest(parent='parent_value')
    page_result = client.list_data_policies(request=request)
    for response in page_result:
        print(response)