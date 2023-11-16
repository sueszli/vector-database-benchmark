from google.maps import mapsplatformdatasets_v1alpha

def sample_delete_dataset():
    if False:
        for i in range(10):
            print('nop')
    client = mapsplatformdatasets_v1alpha.MapsPlatformDatasetsV1AlphaClient()
    request = mapsplatformdatasets_v1alpha.DeleteDatasetRequest(name='name_value')
    client.delete_dataset(request=request)