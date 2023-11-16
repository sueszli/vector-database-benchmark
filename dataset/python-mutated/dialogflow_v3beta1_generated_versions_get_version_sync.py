from google.cloud import dialogflowcx_v3beta1

def sample_get_version():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3beta1.VersionsClient()
    request = dialogflowcx_v3beta1.GetVersionRequest(name='name_value')
    response = client.get_version(request=request)
    print(response)