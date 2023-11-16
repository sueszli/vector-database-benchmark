from google.cloud import bigquery_connection_v1

def sample_list_connections():
    if False:
        i = 10
        return i + 15
    client = bigquery_connection_v1.ConnectionServiceClient()
    request = bigquery_connection_v1.ListConnectionsRequest(parent='parent_value', page_size=951)
    page_result = client.list_connections(request=request)
    for response in page_result:
        print(response)