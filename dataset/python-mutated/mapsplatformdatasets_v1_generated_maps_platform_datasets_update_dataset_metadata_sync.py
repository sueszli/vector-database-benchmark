from google.maps import mapsplatformdatasets_v1

def sample_update_dataset_metadata():
    if False:
        i = 10
        return i + 15
    client = mapsplatformdatasets_v1.MapsPlatformDatasetsClient()
    request = mapsplatformdatasets_v1.UpdateDatasetMetadataRequest()
    response = client.update_dataset_metadata(request=request)
    print(response)