from google.cloud import oslogin_v1

def sample_get_ssh_public_key():
    if False:
        i = 10
        return i + 15
    client = oslogin_v1.OsLoginServiceClient()
    request = oslogin_v1.GetSshPublicKeyRequest(name='name_value')
    response = client.get_ssh_public_key(request=request)
    print(response)