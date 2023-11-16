from google.cloud import api_keys_v2
from google.cloud.api_keys_v2 import Key

def create_api_key(project_id: str, suffix: str) -> Key:
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates and restrict an API key. Add the suffix for uniqueness.\n\n    TODO(Developer):\n    1. Before running this sample,\n      set up ADC as described in https://cloud.google.com/docs/authentication/external/set-up-adc\n    2. Make sure you have the necessary permission to create API keys.\n\n    Args:\n        project_id: Google Cloud project id.\n\n    Returns:\n        response: Returns the created API Key.\n    '
    client = api_keys_v2.ApiKeysClient()
    key = api_keys_v2.Key()
    key.display_name = f'My first API key - {suffix}'
    request = api_keys_v2.CreateKeyRequest()
    request.parent = f'projects/{project_id}/locations/global'
    request.key = key
    response = client.create_key(request=request).result()
    print(f'Successfully created an API key: {response.name}')
    return response