from google.cloud import dataplex_v1

def sample_get_data_taxonomy():
    if False:
        while True:
            i = 10
    client = dataplex_v1.DataTaxonomyServiceClient()
    request = dataplex_v1.GetDataTaxonomyRequest(name='name_value')
    response = client.get_data_taxonomy(request=request)
    print(response)