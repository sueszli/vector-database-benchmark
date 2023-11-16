from google.cloud import bigquery_biglake_v1alpha1

def sample_list_catalogs():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.ListCatalogsRequest(parent='parent_value')
    page_result = client.list_catalogs(request=request)
    for response in page_result:
        print(response)