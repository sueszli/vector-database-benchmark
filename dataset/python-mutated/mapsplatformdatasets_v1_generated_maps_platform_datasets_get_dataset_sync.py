from google.maps import mapsplatformdatasets_v1

def sample_get_dataset():
    if False:
        while True:
            i = 10
    client = mapsplatformdatasets_v1.MapsPlatformDatasetsClient()
    request = mapsplatformdatasets_v1.GetDatasetRequest(name='name_value')
    response = client.get_dataset(request=request)
    print(response)