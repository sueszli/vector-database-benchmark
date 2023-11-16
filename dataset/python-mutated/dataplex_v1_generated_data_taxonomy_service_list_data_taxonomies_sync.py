from google.cloud import dataplex_v1

def sample_list_data_taxonomies():
    if False:
        return 10
    client = dataplex_v1.DataTaxonomyServiceClient()
    request = dataplex_v1.ListDataTaxonomiesRequest(parent='parent_value')
    page_result = client.list_data_taxonomies(request=request)
    for response in page_result:
        print(response)