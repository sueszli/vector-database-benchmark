from google.cloud import vision_v1p3beta1

def sample_get_product_set():
    if False:
        print('Hello World!')
    client = vision_v1p3beta1.ProductSearchClient()
    request = vision_v1p3beta1.GetProductSetRequest(name='name_value')
    response = client.get_product_set(request=request)
    print(response)