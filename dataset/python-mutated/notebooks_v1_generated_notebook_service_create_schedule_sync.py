from google.cloud import notebooks_v1

def sample_create_schedule():
    if False:
        for i in range(10):
            print('nop')
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.CreateScheduleRequest(parent='parent_value', schedule_id='schedule_id_value')
    operation = client.create_schedule(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)