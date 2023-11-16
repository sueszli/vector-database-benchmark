from google.maps import fleetengine_delivery_v1

def sample_batch_create_tasks():
    if False:
        for i in range(10):
            print('nop')
    client = fleetengine_delivery_v1.DeliveryServiceClient()
    requests = fleetengine_delivery_v1.CreateTaskRequest()
    requests.parent = 'parent_value'
    requests.task_id = 'task_id_value'
    requests.task.type_ = 'UNAVAILABLE'
    requests.task.state = 'CLOSED'
    request = fleetengine_delivery_v1.BatchCreateTasksRequest(parent='parent_value', requests=requests)
    response = client.batch_create_tasks(request=request)
    print(response)