from google.cloud import notebooks_v1

def sample_list_schedules():
    if False:
        for i in range(10):
            print('nop')
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.ListSchedulesRequest(parent='parent_value')
    page_result = client.list_schedules(request=request)
    for response in page_result:
        print(response)