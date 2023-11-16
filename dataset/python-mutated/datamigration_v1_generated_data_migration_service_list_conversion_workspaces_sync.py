from google.cloud import clouddms_v1

def sample_list_conversion_workspaces():
    if False:
        for i in range(10):
            print('nop')
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.ListConversionWorkspacesRequest(parent='parent_value')
    page_result = client.list_conversion_workspaces(request=request)
    for response in page_result:
        print(response)