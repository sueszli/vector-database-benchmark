from google.ai import generativelanguage_v1beta3

def sample_delete_permission():
    if False:
        print('Hello World!')
    client = generativelanguage_v1beta3.PermissionServiceClient()
    request = generativelanguage_v1beta3.DeletePermissionRequest(name='name_value')
    client.delete_permission(request=request)