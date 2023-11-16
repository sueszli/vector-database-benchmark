from google.cloud import dialogflowcx_v3

def sample_delete_version():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3.VersionsClient()
    request = dialogflowcx_v3.DeleteVersionRequest(name='name_value')
    client.delete_version(request=request)