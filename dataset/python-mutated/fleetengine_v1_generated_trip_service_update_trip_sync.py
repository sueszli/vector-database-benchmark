from google.maps import fleetengine_v1

def sample_update_trip():
    if False:
        return 10
    client = fleetengine_v1.TripServiceClient()
    request = fleetengine_v1.UpdateTripRequest(name='name_value')
    response = client.update_trip(request=request)
    print(response)