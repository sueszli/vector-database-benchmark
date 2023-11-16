from google.cloud import bigquery_biglake_v1

def sample_create_database():
    if False:
        while True:
            i = 10
    client = bigquery_biglake_v1.MetastoreServiceClient()
    request = bigquery_biglake_v1.CreateDatabaseRequest(parent='parent_value', database_id='database_id_value')
    response = client.create_database(request=request)
    print(response)