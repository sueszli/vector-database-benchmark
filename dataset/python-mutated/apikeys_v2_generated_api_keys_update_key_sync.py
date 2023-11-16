from google.cloud import api_keys_v2

def sample_update_key():
    if False:
        print('Hello World!')
    client = api_keys_v2.ApiKeysClient()
    request = api_keys_v2.UpdateKeyRequest()
    operation = client.update_key(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)