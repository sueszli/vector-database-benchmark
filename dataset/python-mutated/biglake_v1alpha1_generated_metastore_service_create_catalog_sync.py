from google.cloud import bigquery_biglake_v1alpha1

def sample_create_catalog():
    if False:
        return 10
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.CreateCatalogRequest(parent='parent_value', catalog_id='catalog_id_value')
    response = client.create_catalog(request=request)
    print(response)