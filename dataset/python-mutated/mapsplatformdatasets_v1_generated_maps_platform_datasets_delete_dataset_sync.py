from google.maps import mapsplatformdatasets_v1

def sample_delete_dataset():
    if False:
        while True:
            i = 10
    client = mapsplatformdatasets_v1.MapsPlatformDatasetsClient()
    request = mapsplatformdatasets_v1.DeleteDatasetRequest(name='name_value')
    client.delete_dataset(request=request)