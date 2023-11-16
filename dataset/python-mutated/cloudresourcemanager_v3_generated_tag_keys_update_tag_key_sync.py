from google.cloud import resourcemanager_v3

def sample_update_tag_key():
    if False:
        print('Hello World!')
    client = resourcemanager_v3.TagKeysClient()
    tag_key = resourcemanager_v3.TagKey()
    tag_key.short_name = 'short_name_value'
    request = resourcemanager_v3.UpdateTagKeyRequest(tag_key=tag_key)
    operation = client.update_tag_key(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)