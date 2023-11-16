from google.maps import places_v1

def sample_search_nearby():
    if False:
        return 10
    client = places_v1.PlacesClient()
    location_restriction = places_v1.LocationRestriction()
    location_restriction.circle.radius = 0.648
    request = places_v1.SearchNearbyRequest(location_restriction=location_restriction)
    response = client.search_nearby(request=request)
    print(response)