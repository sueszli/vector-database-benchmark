from google.cloud import bigquery_biglake_v1

def sample_delete_database():
    if False:
        print('Hello World!')
    client = bigquery_biglake_v1.MetastoreServiceClient()
    request = bigquery_biglake_v1.DeleteDatabaseRequest(name='name_value')
    response = client.delete_database(request=request)
    print(response)