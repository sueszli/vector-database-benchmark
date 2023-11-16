from google.cloud import gkehub_v1

def sample_create_feature():
    if False:
        print('Hello World!')
    client = gkehub_v1.GkeHubClient()
    request = gkehub_v1.CreateFeatureRequest()
    operation = client.create_feature(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)