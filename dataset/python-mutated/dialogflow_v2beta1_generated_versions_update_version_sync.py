from google.cloud import dialogflow_v2beta1

def sample_update_version():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.VersionsClient()
    request = dialogflow_v2beta1.UpdateVersionRequest()
    response = client.update_version(request=request)
    print(response)