from google.cloud import resourcemanager_v3

def sample_get_tag_key():
    if False:
        i = 10
        return i + 15
    client = resourcemanager_v3.TagKeysClient()
    request = resourcemanager_v3.GetTagKeyRequest(name='name_value')
    response = client.get_tag_key(request=request)
    print(response)