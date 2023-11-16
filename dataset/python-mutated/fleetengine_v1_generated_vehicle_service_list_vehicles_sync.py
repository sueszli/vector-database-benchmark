from google.maps import fleetengine_v1

def sample_list_vehicles():
    if False:
        return 10
    client = fleetengine_v1.VehicleServiceClient()
    request = fleetengine_v1.ListVehiclesRequest(parent='parent_value', vehicle_type_categories=['PEDESTRIAN'])
    page_result = client.list_vehicles(request=request)
    for response in page_result:
        print(response)