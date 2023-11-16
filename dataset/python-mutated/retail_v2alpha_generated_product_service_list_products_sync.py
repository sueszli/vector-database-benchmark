from google.cloud import retail_v2alpha

def sample_list_products():
    if False:
        print('Hello World!')
    client = retail_v2alpha.ProductServiceClient()
    request = retail_v2alpha.ListProductsRequest(parent='parent_value')
    page_result = client.list_products(request=request)
    for response in page_result:
        print(response)