from google.cloud import api_keys_v2

def sample_lookup_key():
    if False:
        while True:
            i = 10
    client = api_keys_v2.ApiKeysClient()
    request = api_keys_v2.LookupKeyRequest(key_string='key_string_value')
    response = client.lookup_key(request=request)
    print(response)