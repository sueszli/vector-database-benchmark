from google.cloud import resourcemanager_v3

def sample_get_tag_value():
    if False:
        return 10
    client = resourcemanager_v3.TagValuesClient()
    request = resourcemanager_v3.GetTagValueRequest(name='name_value')
    response = client.get_tag_value(request=request)
    print(response)