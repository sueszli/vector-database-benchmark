from google.maps import fleetengine_v1

def sample_update_vehicle():
    if False:
        for i in range(10):
            print('nop')
    client = fleetengine_v1.VehicleServiceClient()
    request = fleetengine_v1.UpdateVehicleRequest(name='name_value')
    response = client.update_vehicle(request=request)
    print(response)