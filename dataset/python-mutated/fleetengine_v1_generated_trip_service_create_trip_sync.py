from google.maps import fleetengine_v1

def sample_create_trip():
    if False:
        i = 10
        return i + 15
    client = fleetengine_v1.TripServiceClient()
    request = fleetengine_v1.CreateTripRequest(parent='parent_value', trip_id='trip_id_value')
    response = client.create_trip(request=request)
    print(response)