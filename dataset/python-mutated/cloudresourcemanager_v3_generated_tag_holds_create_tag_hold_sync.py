from google.cloud import resourcemanager_v3

def sample_create_tag_hold():
    if False:
        i = 10
        return i + 15
    client = resourcemanager_v3.TagHoldsClient()
    tag_hold = resourcemanager_v3.TagHold()
    tag_hold.holder = 'holder_value'
    request = resourcemanager_v3.CreateTagHoldRequest(parent='parent_value', tag_hold=tag_hold)
    operation = client.create_tag_hold(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)