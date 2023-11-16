from google.cloud import vision_v1p4beta1

def sample_list_products_in_product_set():
    if False:
        return 10
    client = vision_v1p4beta1.ProductSearchClient()
    request = vision_v1p4beta1.ListProductsInProductSetRequest(name='name_value')
    page_result = client.list_products_in_product_set(request=request)
    for response in page_result:
        print(response)