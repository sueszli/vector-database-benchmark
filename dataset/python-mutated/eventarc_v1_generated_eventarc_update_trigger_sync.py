from google.cloud import eventarc_v1

def sample_update_trigger():
    if False:
        for i in range(10):
            print('nop')
    client = eventarc_v1.EventarcClient()
    request = eventarc_v1.UpdateTriggerRequest(validate_only=True)
    operation = client.update_trigger(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)