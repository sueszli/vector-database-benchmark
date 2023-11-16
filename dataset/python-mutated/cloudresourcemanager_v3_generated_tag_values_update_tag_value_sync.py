from google.cloud import resourcemanager_v3

def sample_update_tag_value():
    if False:
        i = 10
        return i + 15
    client = resourcemanager_v3.TagValuesClient()
    tag_value = resourcemanager_v3.TagValue()
    tag_value.short_name = 'short_name_value'
    request = resourcemanager_v3.UpdateTagValueRequest(tag_value=tag_value)
    operation = client.update_tag_value(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)