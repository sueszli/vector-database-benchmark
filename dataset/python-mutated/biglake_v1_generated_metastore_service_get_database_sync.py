from google.cloud import bigquery_biglake_v1

def sample_get_database():
    if False:
        return 10
    client = bigquery_biglake_v1.MetastoreServiceClient()
    request = bigquery_biglake_v1.GetDatabaseRequest(name='name_value')
    response = client.get_database(request=request)
    print(response)