from google.cloud import resourcemanager_v3

def sample_delete_tag_key():
    if False:
        for i in range(10):
            print('nop')
    client = resourcemanager_v3.TagKeysClient()
    request = resourcemanager_v3.DeleteTagKeyRequest(name='name_value')
    operation = client.delete_tag_key(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)