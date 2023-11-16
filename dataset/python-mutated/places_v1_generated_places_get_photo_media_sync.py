from google.maps import places_v1

def sample_get_photo_media():
    if False:
        for i in range(10):
            print('nop')
    client = places_v1.PlacesClient()
    request = places_v1.GetPhotoMediaRequest(name='name_value')
    response = client.get_photo_media(request=request)
    print(response)