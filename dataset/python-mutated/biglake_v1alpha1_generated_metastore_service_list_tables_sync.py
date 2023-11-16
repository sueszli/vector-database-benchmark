from google.cloud import bigquery_biglake_v1alpha1

def sample_list_tables():
    if False:
        while True:
            i = 10
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.ListTablesRequest(parent='parent_value')
    page_result = client.list_tables(request=request)
    for response in page_result:
        print(response)