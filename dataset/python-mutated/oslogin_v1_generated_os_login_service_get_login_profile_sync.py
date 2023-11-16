from google.cloud import oslogin_v1

def sample_get_login_profile():
    if False:
        print('Hello World!')
    client = oslogin_v1.OsLoginServiceClient()
    request = oslogin_v1.GetLoginProfileRequest(name='name_value')
    response = client.get_login_profile(request=request)
    print(response)