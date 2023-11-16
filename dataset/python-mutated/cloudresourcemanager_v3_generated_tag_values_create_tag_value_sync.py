from google.cloud import resourcemanager_v3

def sample_create_tag_value():
    if False:
        print('Hello World!')
    client = resourcemanager_v3.TagValuesClient()
    tag_value = resourcemanager_v3.TagValue()
    tag_value.short_name = 'short_name_value'
    request = resourcemanager_v3.CreateTagValueRequest(tag_value=tag_value)
    operation = client.create_tag_value(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)