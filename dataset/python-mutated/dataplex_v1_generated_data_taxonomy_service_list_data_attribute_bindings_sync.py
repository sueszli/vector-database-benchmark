from google.cloud import dataplex_v1

def sample_list_data_attribute_bindings():
    if False:
        print('Hello World!')
    client = dataplex_v1.DataTaxonomyServiceClient()
    request = dataplex_v1.ListDataAttributeBindingsRequest(parent='parent_value')
    page_result = client.list_data_attribute_bindings(request=request)
    for response in page_result:
        print(response)