from google.cloud import api_keys_v2

def sample_get_key():
    if False:
        return 10
    client = api_keys_v2.ApiKeysClient()
    request = api_keys_v2.GetKeyRequest(name='name_value')
    response = client.get_key(request=request)
    print(response)