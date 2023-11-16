from google.cloud import oslogin_v1

def sample_delete_posix_account():
    if False:
        i = 10
        return i + 15
    client = oslogin_v1.OsLoginServiceClient()
    request = oslogin_v1.DeletePosixAccountRequest(name='name_value')
    client.delete_posix_account(request=request)