from google.cloud import dialogflow_v2

def sample_get_version():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.VersionsClient()
    request = dialogflow_v2.GetVersionRequest(name='name_value')
    response = client.get_version(request=request)
    print(response)