from google.cloud import gkehub_v1

def sample_update_feature():
    if False:
        while True:
            i = 10
    client = gkehub_v1.GkeHubClient()
    request = gkehub_v1.UpdateFeatureRequest()
    operation = client.update_feature(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)