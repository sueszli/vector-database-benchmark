from google.cloud import bigquery_biglake_v1alpha1

def sample_rename_table():
    if False:
        while True:
            i = 10
    client = bigquery_biglake_v1alpha1.MetastoreServiceClient()
    request = bigquery_biglake_v1alpha1.RenameTableRequest(name='name_value', new_name='new_name_value')
    response = client.rename_table(request=request)
    print(response)