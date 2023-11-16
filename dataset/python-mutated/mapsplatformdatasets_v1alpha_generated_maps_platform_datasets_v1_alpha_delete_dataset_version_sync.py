from google.maps import mapsplatformdatasets_v1alpha

def sample_delete_dataset_version():
    if False:
        print('Hello World!')
    client = mapsplatformdatasets_v1alpha.MapsPlatformDatasetsV1AlphaClient()
    request = mapsplatformdatasets_v1alpha.DeleteDatasetVersionRequest(name='name_value')
    client.delete_dataset_version(request=request)