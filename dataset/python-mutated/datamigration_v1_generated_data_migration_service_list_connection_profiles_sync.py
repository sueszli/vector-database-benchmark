from google.cloud import clouddms_v1

def sample_list_connection_profiles():
    if False:
        print('Hello World!')
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.ListConnectionProfilesRequest(parent='parent_value')
    page_result = client.list_connection_profiles(request=request)
    for response in page_result:
        print(response)