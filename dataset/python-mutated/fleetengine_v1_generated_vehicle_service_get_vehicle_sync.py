from google.maps import fleetengine_v1

def sample_get_vehicle():
    if False:
        print('Hello World!')
    client = fleetengine_v1.VehicleServiceClient()
    request = fleetengine_v1.GetVehicleRequest(name='name_value')
    response = client.get_vehicle(request=request)
    print(response)