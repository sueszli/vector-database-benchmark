from google.cloud import dataform_v1beta1

def sample_list_workspaces():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.ListWorkspacesRequest(parent='parent_value')
    page_result = client.list_workspaces(request=request)
    for response in page_result:
        print(response)