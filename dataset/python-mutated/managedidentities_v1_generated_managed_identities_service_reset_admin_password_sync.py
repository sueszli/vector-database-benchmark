from google.cloud import managedidentities_v1

def sample_reset_admin_password():
    if False:
        print('Hello World!')
    client = managedidentities_v1.ManagedIdentitiesServiceClient()
    request = managedidentities_v1.ResetAdminPasswordRequest(name='name_value')
    response = client.reset_admin_password(request=request)
    print(response)