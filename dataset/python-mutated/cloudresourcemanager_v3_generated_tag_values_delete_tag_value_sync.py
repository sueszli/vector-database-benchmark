from google.cloud import resourcemanager_v3

def sample_delete_tag_value():
    if False:
        print('Hello World!')
    client = resourcemanager_v3.TagValuesClient()
    request = resourcemanager_v3.DeleteTagValueRequest(name='name_value')
    operation = client.delete_tag_value(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)