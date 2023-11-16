from google.cloud import bigquery_biglake_v1alpha1

def sample_update_database():
    if False:
        i = 10
        return i + 15
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.UpdateDatabaseRequest()
    response = client.update_database(request=request)
    print(response)