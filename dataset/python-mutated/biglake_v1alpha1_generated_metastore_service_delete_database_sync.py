from google.cloud import bigquery_biglake_v1alpha1

def sample_delete_database():
    if False:
        return 10
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.DeleteDatabaseRequest(name='name_value')
    response = client.delete_database(request=request)
    print(response)