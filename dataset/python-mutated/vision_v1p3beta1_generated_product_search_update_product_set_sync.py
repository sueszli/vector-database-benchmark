from google.cloud import vision_v1p3beta1

def sample_update_product_set():
    if False:
        i = 10
        return i + 15
    client = vision_v1p3beta1.ProductSearchClient()
    request = vision_v1p3beta1.UpdateProductSetRequest()
    response = client.update_product_set(request=request)
    print(response)