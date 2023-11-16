from google.cloud import bigquery_biglake_v1

def sample_delete_table():
    if False:
        i = 10
        return i + 15
    client = bigquery_biglake_v1.MetastoreServiceClient()
    request = bigquery_biglake_v1.DeleteTableRequest(name='name_value')
    response = client.delete_table(request=request)
    print(response)