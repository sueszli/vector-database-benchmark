from google.cloud import bigquery_biglake_v1alpha1

def sample_get_database():
    if False:
        while True:
            i = 10
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.GetDatabaseRequest(name='name_value')
    response = client.get_database(request=request)
    print(response)