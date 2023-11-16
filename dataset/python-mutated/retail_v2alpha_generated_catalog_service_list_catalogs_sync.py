from google.cloud import retail_v2alpha

def sample_list_catalogs():
    if False:
        while True:
            i = 10
    client = retail_v2alpha.CatalogServiceClient()
    request = retail_v2alpha.ListCatalogsRequest(parent='parent_value')
    page_result = client.list_catalogs(request=request)
    for response in page_result:
        print(response)