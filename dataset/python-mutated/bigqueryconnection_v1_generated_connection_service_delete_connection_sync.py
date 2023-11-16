from google.cloud import bigquery_connection_v1

def sample_delete_connection():
    if False:
        i = 10
        return i + 15
    client = bigquery_connection_v1.ConnectionServiceClient()
    request = bigquery_connection_v1.DeleteConnectionRequest(name='name_value')
    client.delete_connection(request=request)