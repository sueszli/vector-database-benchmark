from google.cloud import vision_v1p4beta1

def sample_list_products():
    if False:
        for i in range(10):
            print('nop')
    client = vision_v1p4beta1.ProductSearchClient()
    request = vision_v1p4beta1.ListProductsRequest(parent='parent_value')
    page_result = client.list_products(request=request)
    for response in page_result:
        print(response)