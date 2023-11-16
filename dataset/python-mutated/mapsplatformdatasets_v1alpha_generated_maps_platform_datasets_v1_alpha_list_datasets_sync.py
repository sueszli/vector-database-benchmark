from google.maps import mapsplatformdatasets_v1alpha

def sample_list_datasets():
    if False:
        print('Hello World!')
    client = mapsplatformdatasets_v1alpha.MapsPlatformDatasetsV1AlphaClient()
    request = mapsplatformdatasets_v1alpha.ListDatasetsRequest(parent='parent_value')
    page_result = client.list_datasets(request=request)
    for response in page_result:
        print(response)