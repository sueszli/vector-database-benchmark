from google.maps import fleetengine_v1

def sample_update_vehicle_attributes():
    if False:
        for i in range(10):
            print('nop')
    client = fleetengine_v1.VehicleServiceClient()
    attributes = fleetengine_v1.VehicleAttribute()
    attributes.string_value = 'string_value_value'
    request = fleetengine_v1.UpdateVehicleAttributesRequest(name='name_value', attributes=attributes)
    response = client.update_vehicle_attributes(request=request)
    print(response)