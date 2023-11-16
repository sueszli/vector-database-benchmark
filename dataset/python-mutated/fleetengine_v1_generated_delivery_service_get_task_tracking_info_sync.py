from google.maps import fleetengine_delivery_v1

def sample_get_task_tracking_info():
    if False:
        return 10
    client = fleetengine_delivery_v1.DeliveryServiceClient()
    request = fleetengine_delivery_v1.GetTaskTrackingInfoRequest(name='name_value')
    response = client.get_task_tracking_info(request=request)
    print(response)