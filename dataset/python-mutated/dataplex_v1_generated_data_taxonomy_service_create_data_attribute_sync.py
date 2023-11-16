from google.cloud import dataplex_v1

def sample_create_data_attribute():
    if False:
        while True:
            i = 10
    client = dataplex_v1.DataTaxonomyServiceClient()
    request = dataplex_v1.CreateDataAttributeRequest(parent='parent_value', data_attribute_id='data_attribute_id_value')
    operation = client.create_data_attribute(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)