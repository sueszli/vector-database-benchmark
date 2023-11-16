from google.cloud import api_keys_v2
from google.cloud.api_keys_v2 import Key

def restrict_api_key_http(project_id: str, key_id: str) -> Key:
    if False:
        i = 10
        return i + 15
    '\n    Restricts an API key. To restrict the websites that can use your API key,\n    you add one or more HTTP referrer restrictions.\n\n    TODO(Developer): Replace the variables before running this sample.\n\n    Args:\n        project_id: Google Cloud project id.\n        key_id: ID of the key to restrict. This ID is auto-created during key creation.\n            This is different from the key string. To obtain the key_id,\n            you can also use the lookup api: client.lookup_key()\n\n    Returns:\n        response: Returns the updated API Key.\n    '
    client = api_keys_v2.ApiKeysClient()
    browser_key_restrictions = api_keys_v2.BrowserKeyRestrictions()
    browser_key_restrictions.allowed_referrers = ['www.example.com/*']
    restrictions = api_keys_v2.Restrictions()
    restrictions.browser_key_restrictions = browser_key_restrictions
    key = api_keys_v2.Key()
    key.name = f'projects/{project_id}/locations/global/keys/{key_id}'
    key.restrictions = restrictions
    request = api_keys_v2.UpdateKeyRequest()
    request.key = key
    request.update_mask = 'restrictions'
    response = client.update_key(request=request).result()
    print(f'Successfully updated the API key: {response.name}')
    return response