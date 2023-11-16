from google.cloud import dialogflow_v2

def sample_create_version():
    if False:
        return 10
    client = dialogflow_v2.VersionsClient()
    request = dialogflow_v2.CreateVersionRequest(parent='parent_value')
    response = client.create_version(request=request)
    print(response)