from google.cloud import dialogflow_v2beta1

def sample_delete_version():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.VersionsClient()
    request = dialogflow_v2beta1.DeleteVersionRequest(name='name_value')
    client.delete_version(request=request)