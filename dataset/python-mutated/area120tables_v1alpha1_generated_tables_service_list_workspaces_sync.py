from google.area120 import tables_v1alpha1

def sample_list_workspaces():
    if False:
        i = 10
        return i + 15
    client = tables_v1alpha1.TablesServiceClient()
    request = tables_v1alpha1.ListWorkspacesRequest()
    page_result = client.list_workspaces(request=request)
    for response in page_result:
        print(response)