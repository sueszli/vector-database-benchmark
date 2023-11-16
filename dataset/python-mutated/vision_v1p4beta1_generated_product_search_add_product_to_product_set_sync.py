from google.cloud import vision_v1p4beta1

def sample_add_product_to_product_set():
    if False:
        print('Hello World!')
    client = vision_v1p4beta1.ProductSearchClient()
    request = vision_v1p4beta1.AddProductToProductSetRequest(name='name_value', product='product_value')
    client.add_product_to_product_set(request=request)