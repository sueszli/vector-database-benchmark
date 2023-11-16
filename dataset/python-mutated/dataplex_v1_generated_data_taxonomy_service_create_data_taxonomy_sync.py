from google.cloud import dataplex_v1

def sample_create_data_taxonomy():
    if False:
        return 10
    client = dataplex_v1.DataTaxonomyServiceClient()
    request = dataplex_v1.CreateDataTaxonomyRequest(parent='parent_value', data_taxonomy_id='data_taxonomy_id_value')
    operation = client.create_data_taxonomy(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)