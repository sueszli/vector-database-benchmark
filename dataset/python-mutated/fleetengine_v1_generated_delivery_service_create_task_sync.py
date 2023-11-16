from google.maps import fleetengine_delivery_v1

def sample_create_task():
    if False:
        i = 10
        return i + 15
    client = fleetengine_delivery_v1.DeliveryServiceClient()
    task = fleetengine_delivery_v1.Task()
    task.type_ = 'UNAVAILABLE'
    task.state = 'CLOSED'
    request = fleetengine_delivery_v1.CreateTaskRequest(parent='parent_value', task_id='task_id_value', task=task)
    response = client.create_task(request=request)
    print(response)