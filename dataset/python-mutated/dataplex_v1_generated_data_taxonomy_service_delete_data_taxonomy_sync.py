from google.cloud import dataplex_v1

def sample_delete_data_taxonomy():
    if False:
        while True:
            i = 10
    client = dataplex_v1.DataTaxonomyServiceClient()
    request = dataplex_v1.DeleteDataTaxonomyRequest(name='name_value')
    operation = client.delete_data_taxonomy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)