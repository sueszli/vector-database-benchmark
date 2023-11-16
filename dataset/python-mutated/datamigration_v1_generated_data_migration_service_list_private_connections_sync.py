from google.cloud import clouddms_v1

def sample_list_private_connections():
    if False:
        i = 10
        return i + 15
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.ListPrivateConnectionsRequest(parent='parent_value')
    page_result = client.list_private_connections(request=request)
    for response in page_result:
        print(response)