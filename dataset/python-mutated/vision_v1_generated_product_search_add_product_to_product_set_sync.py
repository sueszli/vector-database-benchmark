from google.cloud import vision_v1

def sample_add_product_to_product_set():
    if False:
        i = 10
        return i + 15
    client = vision_v1.ProductSearchClient()
    request = vision_v1.AddProductToProductSetRequest(name='name_value', product='product_value')
    client.add_product_to_product_set(request=request)