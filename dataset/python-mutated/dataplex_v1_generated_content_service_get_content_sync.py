from google.cloud import dataplex_v1

def sample_get_content():
    if False:
        return 10
    client = dataplex_v1.ContentServiceClient()
    request = dataplex_v1.GetContentRequest(name='name_value')
    response = client.get_content(request=request)
    print(response)