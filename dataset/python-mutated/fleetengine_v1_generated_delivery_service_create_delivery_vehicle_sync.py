from google.maps import fleetengine_delivery_v1

def sample_create_delivery_vehicle():
    if False:
        while True:
            i = 10
    client = fleetengine_delivery_v1.DeliveryServiceClient()
    request = fleetengine_delivery_v1.CreateDeliveryVehicleRequest(parent='parent_value', delivery_vehicle_id='delivery_vehicle_id_value')
    response = client.create_delivery_vehicle(request=request)
    print(response)