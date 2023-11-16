from google.cloud import bigquery_biglake_v1alpha1

def sample_delete_catalog():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.DeleteCatalogRequest(name='name_value')
    response = client.delete_catalog(request=request)
    print(response)