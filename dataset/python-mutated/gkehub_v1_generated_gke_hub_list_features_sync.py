from google.cloud import gkehub_v1

def sample_list_features():
    if False:
        for i in range(10):
            print('nop')
    client = gkehub_v1.GkeHubClient()
    request = gkehub_v1.ListFeaturesRequest()
    page_result = client.list_features(request=request)
    for response in page_result:
        print(response)