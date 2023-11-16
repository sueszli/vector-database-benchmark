from google.maps import mapsplatformdatasets_v1alpha

def sample_get_dataset():
    if False:
        i = 10
        return i + 15
    client = mapsplatformdatasets_v1alpha.MapsPlatformDatasetsV1AlphaClient()
    request = mapsplatformdatasets_v1alpha.GetDatasetRequest(name='name_value')
    response = client.get_dataset(request=request)
    print(response)