from google.cloud import api_keys_v2

def delete_api_key(project_id: str, key_id: str) -> None:
    if False:
        return 10
    '\n    Deletes an API key.\n\n    TODO(Developer):\n    1. Before running this sample,\n      set up ADC as described in https://cloud.google.com/docs/authentication/external/set-up-adc\n    2. Make sure you have the necessary permission to delete API keys.\n\n    Args:\n        project_id: Google Cloud project id that has the API key to delete.\n        key_id: The API key id to delete.\n    '
    client = api_keys_v2.ApiKeysClient()
    delete_key_request = api_keys_v2.DeleteKeyRequest()
    delete_key_request.name = f'projects/{project_id}/locations/global/keys/{key_id}'
    result = client.delete_key(delete_key_request).result()
    print(f'Successfully deleted the API key: {result.name}')