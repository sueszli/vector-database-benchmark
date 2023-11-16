from google.cloud import dataplex_v1

def sample_update_data_attribute_binding():
    if False:
        for i in range(10):
            print('nop')
    client = dataplex_v1.DataTaxonomyServiceClient()
    data_attribute_binding = dataplex_v1.DataAttributeBinding()
    data_attribute_binding.resource = 'resource_value'
    request = dataplex_v1.UpdateDataAttributeBindingRequest(data_attribute_binding=data_attribute_binding)
    operation = client.update_data_attribute_binding(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)