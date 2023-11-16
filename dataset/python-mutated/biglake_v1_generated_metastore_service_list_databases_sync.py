from google.cloud import bigquery_biglake_v1

def sample_list_databases():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_biglake_v1.MetastoreServiceClient()
    request = bigquery_biglake_v1.ListDatabasesRequest(parent='parent_value')
    page_result = client.list_databases(request=request)
    for response in page_result:
        print(response)