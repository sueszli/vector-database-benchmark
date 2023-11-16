from google.cloud import billing_v1

def sample_list_skus():
    if False:
        return 10
    client = billing_v1.CloudCatalogClient()
    request = billing_v1.ListSkusRequest(parent='parent_value')
    page_result = client.list_skus(request=request)
    for response in page_result:
        print(response)