from google.cloud import resourcemanager_v3

def sample_get_namespaced_tag_key():
    if False:
        for i in range(10):
            print('nop')
    client = resourcemanager_v3.TagKeysClient()
    request = resourcemanager_v3.GetNamespacedTagKeyRequest(name='name_value')
    response = client.get_namespaced_tag_key(request=request)
    print(response)