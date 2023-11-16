from google.cloud import bigquery_biglake_v1alpha1

def sample_get_table():
    if False:
        return 10
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.GetTableRequest(name='name_value')
    response = client.get_table(request=request)
    print(response)