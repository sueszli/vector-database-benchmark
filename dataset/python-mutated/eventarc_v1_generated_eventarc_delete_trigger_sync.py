from google.cloud import eventarc_v1

def sample_delete_trigger():
    if False:
        while True:
            i = 10
    client = eventarc_v1.EventarcClient()
    request = eventarc_v1.DeleteTriggerRequest(name='name_value', validate_only=True)
    operation = client.delete_trigger(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)