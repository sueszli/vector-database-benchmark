from google.cloud import api_keys_v2
from google.cloud.api_keys_v2 import Key

def restrict_api_key_android(project_id: str, key_id: str) -> Key:
    if False:
        for i in range(10):
            print('nop')
    '\n    Restricts an API key based on android applications.\n\n    Specifies the Android application that can use the key.\n\n    TODO(Developer): Replace the variables before running this sample.\n\n    Args:\n        project_id: Google Cloud project id.\n        key_id: ID of the key to restrict. This ID is auto-created during key creation.\n            This is different from the key string. To obtain the key_id,\n            you can also use the lookup api: client.lookup_key()\n\n    Returns:\n        response: Returns the updated API Key.\n    '
    client = api_keys_v2.ApiKeysClient()
    allowed_application = api_keys_v2.AndroidApplication()
    allowed_application.package_name = 'com.google.appname'
    allowed_application.sha1_fingerprint = '0873D391E987982FBBD30873D391E987982FBBD3'
    android_key_restriction = api_keys_v2.AndroidKeyRestrictions()
    android_key_restriction.allowed_applications = [allowed_application]
    restrictions = api_keys_v2.Restrictions()
    restrictions.android_key_restrictions = android_key_restriction
    key = api_keys_v2.Key()
    key.name = f'projects/{project_id}/locations/global/keys/{key_id}'
    key.restrictions = restrictions
    request = api_keys_v2.UpdateKeyRequest()
    request.key = key
    request.update_mask = 'restrictions'
    response = client.update_key(request=request).result()
    print(f'Successfully updated the API key: {response.name}')
    return response