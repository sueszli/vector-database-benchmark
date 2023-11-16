from google.cloud import vision_v1

def sample_list_products_in_product_set():
    if False:
        while True:
            i = 10
    client = vision_v1.ProductSearchClient()
    request = vision_v1.ListProductsInProductSetRequest(name='name_value')
    page_result = client.list_products_in_product_set(request=request)
    for response in page_result:
        print(response)