from google.cloud import resourcemanager_v3

def sample_create_tag_binding():
    if False:
        print('Hello World!')
    client = resourcemanager_v3.TagBindingsClient()
    request = resourcemanager_v3.CreateTagBindingRequest()
    operation = client.create_tag_binding(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)