from google.cloud import bigquery_biglake_v1

def sample_get_catalog():
    if False:
        i = 10
        return i + 15
    client = bigquery_biglake_v1.MetastoreServiceClient()
    request = bigquery_biglake_v1.GetCatalogRequest(name='name_value')
    response = client.get_catalog(request=request)
    print(response)