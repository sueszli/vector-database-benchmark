from google.cloud import dataplex_v1

def sample_list_data_attributes():
    if False:
        for i in range(10):
            print('nop')
    client = dataplex_v1.DataTaxonomyServiceClient()
    request = dataplex_v1.ListDataAttributesRequest(parent='parent_value')
    page_result = client.list_data_attributes(request=request)
    for response in page_result:
        print(response)