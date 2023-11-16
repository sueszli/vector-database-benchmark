from google.cloud import tasks_v2beta2

def sample_renew_lease():
    if False:
        while True:
            i = 10
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.RenewLeaseRequest(name='name_value')
    response = client.renew_lease(request=request)
    print(response)