from google.maps import mapsplatformdatasets_v1

def sample_create_dataset():
    if False:
        for i in range(10):
            print('nop')
    client = mapsplatformdatasets_v1.MapsPlatformDatasetsClient()
    request = mapsplatformdatasets_v1.CreateDatasetRequest(parent='parent_value')
    response = client.create_dataset(request=request)
    print(response)