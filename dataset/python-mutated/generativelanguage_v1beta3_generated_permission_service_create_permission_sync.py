from google.ai import generativelanguage_v1beta3

def sample_create_permission():
    if False:
        i = 10
        return i + 15
    client = generativelanguage_v1beta3.PermissionServiceClient()
    request = generativelanguage_v1beta3.CreatePermissionRequest(parent='parent_value')
    response = client.create_permission(request=request)
    print(response)