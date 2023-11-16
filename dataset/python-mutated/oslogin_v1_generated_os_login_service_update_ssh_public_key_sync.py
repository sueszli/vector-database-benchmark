from google.cloud import oslogin_v1

def sample_update_ssh_public_key():
    if False:
        while True:
            i = 10
    client = oslogin_v1.OsLoginServiceClient()
    request = oslogin_v1.UpdateSshPublicKeyRequest(name='name_value')
    response = client.update_ssh_public_key(request=request)
    print(response)