from google.cloud import bigquery_biglake_v1

def sample_update_table():
    if False:
        i = 10
        return i + 15
    client = bigquery_biglake_v1.MetastoreServiceClient()
    request = bigquery_biglake_v1.UpdateTableRequest()
    response = client.update_table(request=request)
    print(response)