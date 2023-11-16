from google.maps import mapsplatformdatasets_v1alpha

def sample_update_dataset_metadata():
    if False:
        return 10
    client = mapsplatformdatasets_v1alpha.MapsPlatformDatasetsV1AlphaClient()
    request = mapsplatformdatasets_v1alpha.UpdateDatasetMetadataRequest()
    response = client.update_dataset_metadata(request=request)
    print(response)