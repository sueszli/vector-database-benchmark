from google.cloud import resourcemanager_v3

def sample_delete_tag_binding():
    if False:
        return 10
    client = resourcemanager_v3.TagBindingsClient()
    request = resourcemanager_v3.DeleteTagBindingRequest(name='name_value')
    operation = client.delete_tag_binding(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)