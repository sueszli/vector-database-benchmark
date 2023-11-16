from google.cloud import dataplex_v1

def sample_update_data_attribute():
    if False:
        return 10
    client = dataplex_v1.DataTaxonomyServiceClient()
    request = dataplex_v1.UpdateDataAttributeRequest()
    operation = client.update_data_attribute(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)