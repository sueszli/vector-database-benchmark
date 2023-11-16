from google.cloud import bigquery_biglake_v1

def sample_delete_catalog():
    if False:
        return 10
    client = bigquery_biglake_v1.MetastoreServiceClient()
    request = bigquery_biglake_v1.DeleteCatalogRequest(name='name_value')
    response = client.delete_catalog(request=request)
    print(response)