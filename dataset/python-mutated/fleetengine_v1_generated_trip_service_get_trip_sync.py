from google.maps import fleetengine_v1

def sample_get_trip():
    if False:
        for i in range(10):
            print('nop')
    client = fleetengine_v1.TripServiceClient()
    request = fleetengine_v1.GetTripRequest(name='name_value')
    response = client.get_trip(request=request)
    print(response)