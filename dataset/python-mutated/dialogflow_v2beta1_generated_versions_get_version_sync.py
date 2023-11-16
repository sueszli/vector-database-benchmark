from google.cloud import dialogflow_v2beta1

def sample_get_version():
    if False:
        return 10
    client = dialogflow_v2beta1.VersionsClient()
    request = dialogflow_v2beta1.GetVersionRequest(name='name_value')
    response = client.get_version(request=request)
    print(response)