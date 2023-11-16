from google.cloud import tasks_v2beta3

def sample_list_queues():
    if False:
        i = 10
        return i + 15
    client = tasks_v2beta3.CloudTasksClient()
    request = tasks_v2beta3.ListQueuesRequest(parent='parent_value')
    page_result = client.list_queues(request=request)
    for response in page_result:
        print(response)