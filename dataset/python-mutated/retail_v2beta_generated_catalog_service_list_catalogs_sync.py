from google.cloud import retail_v2beta

def sample_list_catalogs():
    if False:
        while True:
            i = 10
    client = retail_v2beta.CatalogServiceClient()
    request = retail_v2beta.ListCatalogsRequest(parent='parent_value')
    page_result = client.list_catalogs(request=request)
    for response in page_result:
        print(response)