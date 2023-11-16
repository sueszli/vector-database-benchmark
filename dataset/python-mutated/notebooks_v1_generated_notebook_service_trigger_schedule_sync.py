from google.cloud import notebooks_v1

def sample_trigger_schedule():
    if False:
        for i in range(10):
            print('nop')
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.TriggerScheduleRequest(name='name_value')
    operation = client.trigger_schedule(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)