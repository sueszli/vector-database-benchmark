from google.cloud import retail_v2

def sample_list_catalogs():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2.CatalogServiceClient()
    request = retail_v2.ListCatalogsRequest(parent='parent_value')
    page_result = client.list_catalogs(request=request)
    for response in page_result:
        print(response)