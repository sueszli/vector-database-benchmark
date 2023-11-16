from google.cloud import retail_v2beta

def sample_list_products():
    if False:
        print('Hello World!')
    client = retail_v2beta.ProductServiceClient()
    request = retail_v2beta.ListProductsRequest(parent='parent_value')
    page_result = client.list_products(request=request)
    for response in page_result:
        print(response)