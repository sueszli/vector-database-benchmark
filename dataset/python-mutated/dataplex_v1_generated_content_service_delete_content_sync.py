from google.cloud import dataplex_v1

def sample_delete_content():
    if False:
        i = 10
        return i + 15
    client = dataplex_v1.ContentServiceClient()
    request = dataplex_v1.DeleteContentRequest(name='name_value')
    client.delete_content(request=request)