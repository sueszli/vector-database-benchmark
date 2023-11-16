from google.cloud import resourcemanager_v3

def sample_get_namespaced_tag_value():
    if False:
        print('Hello World!')
    client = resourcemanager_v3.TagValuesClient()
    request = resourcemanager_v3.GetNamespacedTagValueRequest(name='name_value')
    response = client.get_namespaced_tag_value(request=request)
    print(response)