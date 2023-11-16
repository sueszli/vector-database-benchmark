from google.cloud import bigquery_biglake_v1alpha1

def sample_get_catalog():
    if False:
        while True:
            i = 10
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.GetCatalogRequest(name='name_value')
    response = client.get_catalog(request=request)
    print(response)