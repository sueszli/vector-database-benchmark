from google.cloud import dataplex_v1

def sample_create_data_attribute_binding():
    if False:
        i = 10
        return i + 15
    client = dataplex_v1.DataTaxonomyServiceClient()
    data_attribute_binding = dataplex_v1.DataAttributeBinding()
    data_attribute_binding.resource = 'resource_value'
    request = dataplex_v1.CreateDataAttributeBindingRequest(parent='parent_value', data_attribute_binding_id='data_attribute_binding_id_value', data_attribute_binding=data_attribute_binding)
    operation = client.create_data_attribute_binding(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)