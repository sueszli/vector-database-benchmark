from google.maps import mapsplatformdatasets_v1alpha

def sample_list_dataset_versions():
    if False:
        i = 10
        return i + 15
    client = mapsplatformdatasets_v1alpha.MapsPlatformDatasetsV1AlphaClient()
    request = mapsplatformdatasets_v1alpha.ListDatasetVersionsRequest(name='name_value')
    page_result = client.list_dataset_versions(request=request)
    for response in page_result:
        print(response)