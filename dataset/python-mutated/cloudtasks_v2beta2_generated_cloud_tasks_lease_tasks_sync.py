from google.cloud import tasks_v2beta2

def sample_lease_tasks():
    if False:
        return 10
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.LeaseTasksRequest(parent='parent_value')
    response = client.lease_tasks(request=request)
    print(response)