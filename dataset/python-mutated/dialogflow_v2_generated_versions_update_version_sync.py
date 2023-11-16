from google.cloud import dialogflow_v2

def sample_update_version():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.VersionsClient()
    request = dialogflow_v2.UpdateVersionRequest()
    response = client.update_version(request=request)
    print(response)