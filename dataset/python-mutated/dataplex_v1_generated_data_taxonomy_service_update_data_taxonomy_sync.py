from google.cloud import dataplex_v1

def sample_update_data_taxonomy():
    if False:
        for i in range(10):
            print('nop')
    client = dataplex_v1.DataTaxonomyServiceClient()
    request = dataplex_v1.UpdateDataTaxonomyRequest()
    operation = client.update_data_taxonomy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)