from google.maps import mapsplatformdatasets_v1

def sample_list_datasets():
    if False:
        while True:
            i = 10
    client = mapsplatformdatasets_v1.MapsPlatformDatasetsClient()
    request = mapsplatformdatasets_v1.ListDatasetsRequest(parent='parent_value')
    page_result = client.list_datasets(request=request)
    for response in page_result:
        print(response)