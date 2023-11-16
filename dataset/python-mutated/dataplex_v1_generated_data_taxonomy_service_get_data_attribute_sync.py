from google.cloud import dataplex_v1

def sample_get_data_attribute():
    if False:
        while True:
            i = 10
    client = dataplex_v1.DataTaxonomyServiceClient()
    request = dataplex_v1.GetDataAttributeRequest(name='name_value')
    response = client.get_data_attribute(request=request)
    print(response)