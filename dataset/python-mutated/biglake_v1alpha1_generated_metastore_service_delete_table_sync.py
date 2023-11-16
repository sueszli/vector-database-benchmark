from google.cloud import bigquery_biglake_v1alpha1

def sample_delete_table():
    if False:
        print('Hello World!')
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.DeleteTableRequest(name='name_value')
    response = client.delete_table(request=request)
    print(response)