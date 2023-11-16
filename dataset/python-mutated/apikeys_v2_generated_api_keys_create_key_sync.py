from google.cloud import api_keys_v2

def sample_create_key():
    if False:
        while True:
            i = 10
    client = api_keys_v2.ApiKeysClient()
    request = api_keys_v2.CreateKeyRequest(parent='parent_value')
    operation = client.create_key(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)