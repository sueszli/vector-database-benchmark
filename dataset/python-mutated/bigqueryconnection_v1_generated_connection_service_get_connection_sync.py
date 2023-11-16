from google.cloud import bigquery_connection_v1

def sample_get_connection():
    if False:
        while True:
            i = 10
    client = bigquery_connection_v1.ConnectionServiceClient()
    request = bigquery_connection_v1.GetConnectionRequest(name='name_value')
    response = client.get_connection(request=request)
    print(response)