from google.cloud import bigquery_biglake_v1

def sample_create_catalog():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_biglake_v1.MetastoreServiceClient()
    request = bigquery_biglake_v1.CreateCatalogRequest(parent='parent_value', catalog_id='catalog_id_value')
    response = client.create_catalog(request=request)
    print(response)