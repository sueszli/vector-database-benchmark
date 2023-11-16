from google.maps import fleetengine_delivery_v1

def sample_get_delivery_vehicle():
    if False:
        while True:
            i = 10
    client = fleetengine_delivery_v1.DeliveryServiceClient()
    request = fleetengine_delivery_v1.GetDeliveryVehicleRequest(name='name_value')
    response = client.get_delivery_vehicle(request=request)
    print(response)