from google.cloud import bigquery_connection_v1

def sample_update_connection():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_connection_v1.ConnectionServiceClient()
    request = bigquery_connection_v1.UpdateConnectionRequest(name='name_value')
    response = client.update_connection(request=request)
    print(response)