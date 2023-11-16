from google.cloud import bigquery_biglake_v1alpha1

def sample_create_database():
    if False:
        print('Hello World!')
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.CreateDatabaseRequest(parent='parent_value', database_id='database_id_value')
    response = client.create_database(request=request)
    print(response)