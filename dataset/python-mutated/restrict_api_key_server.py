from google.cloud import api_keys_v2
from google.cloud.api_keys_v2 import Key

def restrict_api_key_server(project_id: str, key_id: str) -> Key:
    if False:
        for i in range(10):
            print('nop')
    '\n    Restricts the API key based on IP addresses. You can specify one or more IP addresses of the callers,\n    for example web servers or cron jobs, that are allowed to use your API key.\n\n    TODO(Developer): Replace the variables before running this sample.\n\n    Args:\n        project_id: Google Cloud project id.\n        key_id: ID of the key to restrict. This ID is auto-created during key creation.\n            This is different from the key string. To obtain the key_id,\n            you can also use the lookup api: client.lookup_key()\n\n    Returns:\n        response: Returns the updated API Key.\n    '
    client = api_keys_v2.ApiKeysClient()
    server_key_restrictions = api_keys_v2.ServerKeyRestrictions()
    server_key_restrictions.allowed_ips = ['198.51.100.0/24', '2000:db8::/64']
    restrictions = api_keys_v2.Restrictions()
    restrictions.server_key_restrictions = server_key_restrictions
    key = api_keys_v2.Key()
    key.name = f'projects/{project_id}/locations/global/keys/{key_id}'
    key.restrictions = restrictions
    request = api_keys_v2.UpdateKeyRequest()
    request.key = key
    request.update_mask = 'restrictions'
    response = client.update_key(request=request).result()
    print(f'Successfully updated the API key: {response.name}')
    return response