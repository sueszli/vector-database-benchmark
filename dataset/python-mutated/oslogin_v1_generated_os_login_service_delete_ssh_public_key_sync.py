from google.cloud import oslogin_v1

def sample_delete_ssh_public_key():
    if False:
        print('Hello World!')
    client = oslogin_v1.OsLoginServiceClient()
    request = oslogin_v1.DeleteSshPublicKeyRequest(name='name_value')
    client.delete_ssh_public_key(request=request)