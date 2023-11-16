from google.cloud import tasks_v2beta2

def sample_list_queues():
    if False:
        for i in range(10):
            print('nop')
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.ListQueuesRequest(parent='parent_value')
    page_result = client.list_queues(request=request)
    for response in page_result:
        print(response)