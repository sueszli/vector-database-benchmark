from google.maps import fleetengine_delivery_v1

def sample_update_task():
    if False:
        while True:
            i = 10
    client = fleetengine_delivery_v1.DeliveryServiceClient()
    task = fleetengine_delivery_v1.Task()
    task.type_ = 'UNAVAILABLE'
    task.state = 'CLOSED'
    request = fleetengine_delivery_v1.UpdateTaskRequest(task=task)
    response = client.update_task(request=request)
    print(response)