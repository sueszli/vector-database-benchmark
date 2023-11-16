from google.cloud import bigquery_connection_v1

def sample_create_connection():
    if False:
        print('Hello World!')
    client = bigquery_connection_v1.ConnectionServiceClient()
    request = bigquery_connection_v1.CreateConnectionRequest(parent='parent_value')
    response = client.create_connection(request=request)
    print(response)