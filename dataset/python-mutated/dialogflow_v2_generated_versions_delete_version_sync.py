from google.cloud import dialogflow_v2

def sample_delete_version():
    if False:
        return 10
    client = dialogflow_v2.VersionsClient()
    request = dialogflow_v2.DeleteVersionRequest(name='name_value')
    client.delete_version(request=request)