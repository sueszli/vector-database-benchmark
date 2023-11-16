from google.ai import generativelanguage_v1beta3

def sample_get_permission():
    if False:
        i = 10
        return i + 15
    client = generativelanguage_v1beta3.PermissionServiceClient()
    request = generativelanguage_v1beta3.GetPermissionRequest(name='name_value')
    response = client.get_permission(request=request)
    print(response)