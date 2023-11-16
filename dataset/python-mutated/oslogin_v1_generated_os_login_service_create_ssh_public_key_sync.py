from google.cloud import oslogin_v1

def sample_create_ssh_public_key():
    if False:
        for i in range(10):
            print('nop')
    client = oslogin_v1.OsLoginServiceClient()
    request = oslogin_v1.CreateSshPublicKeyRequest(parent='parent_value')
    response = client.create_ssh_public_key(request=request)
    print(response)