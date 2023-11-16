from google.cloud import datacatalog_v1beta1

def sample_list_tags():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1beta1.DataCatalogClient()
    request = datacatalog_v1beta1.ListTagsRequest(parent='parent_value')
    page_result = client.list_tags(request=request)
    for response in page_result:
        print(response)