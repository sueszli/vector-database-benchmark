from google.cloud import tasks_v2beta2

def sample_cancel_lease():
    if False:
        i = 10
        return i + 15
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.CancelLeaseRequest(name='name_value')
    response = client.cancel_lease(request=request)
    print(response)