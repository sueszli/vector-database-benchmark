from google.cloud import resourcemanager_v3

def sample_delete_tag_hold():
    if False:
        print('Hello World!')
    client = resourcemanager_v3.TagHoldsClient()
    request = resourcemanager_v3.DeleteTagHoldRequest(name='name_value')
    operation = client.delete_tag_hold(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)