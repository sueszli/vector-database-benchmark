from google.maps import fleetengine_delivery_v1

def sample_get_task():
    if False:
        i = 10
        return i + 15
    client = fleetengine_delivery_v1.DeliveryServiceClient()
    request = fleetengine_delivery_v1.GetTaskRequest(name='name_value')
    response = client.get_task(request=request)
    print(response)