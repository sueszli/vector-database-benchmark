from google.cloud import gkehub_v1

def sample_delete_feature():
    if False:
        for i in range(10):
            print('nop')
    client = gkehub_v1.GkeHubClient()
    request = gkehub_v1.DeleteFeatureRequest()
    operation = client.delete_feature(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)