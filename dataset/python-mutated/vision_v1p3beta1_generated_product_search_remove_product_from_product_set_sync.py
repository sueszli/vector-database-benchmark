from google.cloud import vision_v1p3beta1

def sample_remove_product_from_product_set():
    if False:
        print('Hello World!')
    client = vision_v1p3beta1.ProductSearchClient()
    request = vision_v1p3beta1.RemoveProductFromProductSetRequest(name='name_value', product='product_value')
    client.remove_product_from_product_set(request=request)