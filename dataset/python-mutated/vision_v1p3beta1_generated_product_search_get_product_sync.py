from google.cloud import vision_v1p3beta1

def sample_get_product():
    if False:
        i = 10
        return i + 15
    client = vision_v1p3beta1.ProductSearchClient()
    request = vision_v1p3beta1.GetProductRequest(name='name_value')
    response = client.get_product(request=request)
    print(response)