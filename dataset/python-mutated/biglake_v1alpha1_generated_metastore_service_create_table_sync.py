from google.cloud import bigquery_biglake_v1alpha1

def sample_create_table():
    if False:
        print('Hello World!')
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.CreateTableRequest(parent='parent_value', table_id='table_id_value')
    response = client.create_table(request=request)
    print(response)