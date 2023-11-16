from google.maps import fleetengine_v1

def sample_update_vehicle_location():
    if False:
        i = 10
        return i + 15
    client = fleetengine_v1.VehicleServiceClient()
    request = fleetengine_v1.UpdateVehicleLocationRequest(name='name_value')
    response = client.update_vehicle_location(request=request)
    print(response)