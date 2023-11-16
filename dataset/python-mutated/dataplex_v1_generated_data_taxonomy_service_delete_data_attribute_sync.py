from google.cloud import dataplex_v1

def sample_delete_data_attribute():
    if False:
        for i in range(10):
            print('nop')
    client = dataplex_v1.DataTaxonomyServiceClient()
    request = dataplex_v1.DeleteDataAttributeRequest(name='name_value')
    operation = client.delete_data_attribute(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)