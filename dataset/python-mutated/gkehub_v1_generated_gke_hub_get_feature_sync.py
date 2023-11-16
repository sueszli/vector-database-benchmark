from google.cloud import gkehub_v1

def sample_get_feature():
    if False:
        print('Hello World!')
    client = gkehub_v1.GkeHubClient()
    request = gkehub_v1.GetFeatureRequest()
    response = client.get_feature(request=request)
    print(response)