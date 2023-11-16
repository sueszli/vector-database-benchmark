from google.cloud import api_keys_v2

def sample_undelete_key():
    if False:
        for i in range(10):
            print('nop')
    client = api_keys_v2.ApiKeysClient()
    request = api_keys_v2.UndeleteKeyRequest(name='name_value')
    operation = client.undelete_key(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)