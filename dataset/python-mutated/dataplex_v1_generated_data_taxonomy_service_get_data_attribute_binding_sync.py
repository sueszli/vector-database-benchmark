from google.cloud import dataplex_v1

def sample_get_data_attribute_binding():
    if False:
        while True:
            i = 10
    client = dataplex_v1.DataTaxonomyServiceClient()
    request = dataplex_v1.GetDataAttributeBindingRequest(name='name_value')
    response = client.get_data_attribute_binding(request=request)
    print(response)