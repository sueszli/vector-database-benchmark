from google.maps import fleetengine_v1

def sample_search_fuzzed_vehicles():
    if False:
        print('Hello World!')
    client = fleetengine_v1.VehicleServiceClient()
    request = fleetengine_v1.SearchVehiclesRequest(parent='parent_value', pickup_radius_meters=2146, count=553, minimum_capacity=1705, trip_types=['EXCLUSIVE'], order_by='COST')
    response = client.search_fuzzed_vehicles(request=request)
    print(response)