from google.cloud import notebooks_v1

def sample_get_schedule():
    if False:
        print('Hello World!')
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.GetScheduleRequest(name='name_value')
    response = client.get_schedule(request=request)
    print(response)