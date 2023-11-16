from google.cloud import vision_v1p4beta1

def sample_get_product_set():
    if False:
        return 10
    client = vision_v1p4beta1.ProductSearchClient()
    request = vision_v1p4beta1.GetProductSetRequest(name='name_value')
    response = client.get_product_set(request=request)
    print(response)