from google.cloud import dialogflow_v2beta1

def sample_create_version():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.VersionsClient()
    request = dialogflow_v2beta1.CreateVersionRequest(parent='parent_value')
    response = client.create_version(request=request)
    print(response)