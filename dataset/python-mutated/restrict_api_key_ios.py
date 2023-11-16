from google.cloud import api_keys_v2
from google.cloud.api_keys_v2 import Key

def restrict_api_key_ios(project_id: str, key_id: str) -> Key:
    if False:
        print('Hello World!')
    '\n    Restricts an API key. You can restrict usage of an API key to specific iOS apps\n    by providing the bundle ID of each app.\n\n    TODO(Developer): Replace the variables before running this sample.\n\n    Args:\n        project_id: Google Cloud project id.\n        key_id: ID of the key to restrict. This ID is auto-created during key creation.\n            This is different from the key string. To obtain the key_id,\n            you can also use the lookup api: client.lookup_key()\n\n    Returns:\n        response: Returns the updated API Key.\n    '
    client = api_keys_v2.ApiKeysClient()
    ios_key_restrictions = api_keys_v2.IosKeyRestrictions()
    ios_key_restrictions.allowed_bundle_ids = ['com.google.gmail', 'com.google.drive']
    restrictions = api_keys_v2.Restrictions()
    restrictions.ios_key_restrictions = ios_key_restrictions
    key = api_keys_v2.Key()
    key.name = f'projects/{project_id}/locations/global/keys/{key_id}'
    key.restrictions = restrictions
    request = api_keys_v2.UpdateKeyRequest()
    request.key = key
    request.update_mask = 'restrictions'
    response = client.update_key(request=request).result()
    print(f'Successfully updated the API key: {response.name}')
    return response