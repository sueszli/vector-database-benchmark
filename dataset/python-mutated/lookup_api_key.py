from google.cloud import api_keys_v2

def lookup_api_key(api_key_string: str) -> None:
    if False:
        return 10
    '\n    Retrieves name (full path) of an API key using the API key string.\n\n    TODO(Developer):\n    1. Before running this sample,\n      set up ADC as described in https://cloud.google.com/docs/authentication/external/set-up-adc\n    2. Make sure you have the necessary permission to view API keys.\n\n    Args:\n        api_key_string: API key string to retrieve the API key name.\n    '
    client = api_keys_v2.ApiKeysClient()
    lookup_key_request = api_keys_v2.LookupKeyRequest(key_string=api_key_string)
    lookup_key_response = client.lookup_key(lookup_key_request)
    print(f'Successfully retrieved the API key name: {lookup_key_response.name}')