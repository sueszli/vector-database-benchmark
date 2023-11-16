from google.maps import places_v1

def sample_get_place():
    if False:
        return 10
    client = places_v1.PlacesClient()
    request = places_v1.GetPlaceRequest(name='name_value')
    response = client.get_place(request=request)
    print(response)