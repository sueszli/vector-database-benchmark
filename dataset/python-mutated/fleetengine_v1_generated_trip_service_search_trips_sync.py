from google.maps import fleetengine_v1

def sample_search_trips():
    if False:
        while True:
            i = 10
    client = fleetengine_v1.TripServiceClient()
    request = fleetengine_v1.SearchTripsRequest(parent='parent_value')
    page_result = client.search_trips(request=request)
    for response in page_result:
        print(response)