from google.cloud import retail_v2

def sample_list_products():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2.ProductServiceClient()
    request = retail_v2.ListProductsRequest(parent='parent_value')
    page_result = client.list_products(request=request)
    for response in page_result:
        print(response)