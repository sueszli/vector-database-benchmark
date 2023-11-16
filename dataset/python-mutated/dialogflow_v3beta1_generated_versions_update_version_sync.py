from google.cloud import dialogflowcx_v3beta1

def sample_update_version():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3beta1.VersionsClient()
    version = dialogflowcx_v3beta1.Version()
    version.display_name = 'display_name_value'
    request = dialogflowcx_v3beta1.UpdateVersionRequest(version=version)
    response = client.update_version(request=request)
    print(response)