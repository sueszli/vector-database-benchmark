from google.cloud import bigquery_biglake_v1alpha1

def sample_list_databases():
    if False:
        while True:
            i = 10
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.ListDatabasesRequest(parent='parent_value')
    page_result = client.list_databases(request=request)
    for response in page_result:
        print(response)