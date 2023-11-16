from google.cloud import vision_v1p3beta1

def sample_list_products():
    if False:
        print('Hello World!')
    client = vision_v1p3beta1.ProductSearchClient()
    request = vision_v1p3beta1.ListProductsRequest(parent='parent_value')
    page_result = client.list_products(request=request)
    for response in page_result:
        print(response)