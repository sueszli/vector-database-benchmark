from google.cloud import dialogflowcx_v3

def sample_get_version():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.VersionsClient()
    request = dialogflowcx_v3.GetVersionRequest(name='name_value')
    response = client.get_version(request=request)
    print(response)