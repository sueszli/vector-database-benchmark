from google.ai import generativelanguage_v1beta3

def sample_list_permissions():
    if False:
        return 10
    client = generativelanguage_v1beta3.PermissionServiceClient()
    request = generativelanguage_v1beta3.ListPermissionsRequest(parent='parent_value')
    page_result = client.list_permissions(request=request)
    for response in page_result:
        print(response)