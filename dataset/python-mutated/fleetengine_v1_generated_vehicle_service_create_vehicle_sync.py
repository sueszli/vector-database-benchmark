from google.maps import fleetengine_v1

def sample_create_vehicle():
    if False:
        for i in range(10):
            print('nop')
    client = fleetengine_v1.VehicleServiceClient()
    request = fleetengine_v1.CreateVehicleRequest(parent='parent_value', vehicle_id='vehicle_id_value')
    response = client.create_vehicle(request=request)
    print(response)