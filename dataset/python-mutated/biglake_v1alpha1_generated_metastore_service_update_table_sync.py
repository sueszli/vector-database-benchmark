from google.cloud import bigquery_biglake_v1alpha1

def sample_update_table():
    if False:
        print('Hello World!')
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.UpdateTableRequest()
    response = client.update_table(request=request)
    print(response)