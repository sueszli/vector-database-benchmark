from google.cloud import dialogflowcx_v3

def sample_update_version():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3.VersionsClient()
    version = dialogflowcx_v3.Version()
    version.display_name = 'display_name_value'
    request = dialogflowcx_v3.UpdateVersionRequest(version=version)
    response = client.update_version(request=request)
    print(response)