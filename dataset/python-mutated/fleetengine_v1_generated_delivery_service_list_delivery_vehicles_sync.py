from google.maps import fleetengine_delivery_v1

def sample_list_delivery_vehicles():
    if False:
        i = 10
        return i + 15
    client = fleetengine_delivery_v1.DeliveryServiceClient()
    request = fleetengine_delivery_v1.ListDeliveryVehiclesRequest(parent='parent_value')
    page_result = client.list_delivery_vehicles(request=request)
    for response in page_result:
        print(response)