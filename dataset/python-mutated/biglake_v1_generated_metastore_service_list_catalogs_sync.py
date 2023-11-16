from google.cloud import bigquery_biglake_v1

def sample_list_catalogs():
    if False:
        return 10
    client = bigquery_biglake_v1.MetastoreServiceClient()
    request = bigquery_biglake_v1.ListCatalogsRequest(parent='parent_value')
    page_result = client.list_catalogs(request=request)
    for response in page_result:
        print(response)