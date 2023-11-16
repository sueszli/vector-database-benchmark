from google.ai import generativelanguage_v1beta3

def sample_update_permission():
    if False:
        while True:
            i = 10
    client = generativelanguage_v1beta3.PermissionServiceClient()
    request = generativelanguage_v1beta3.UpdatePermissionRequest()
    response = client.update_permission(request=request)
    print(response)