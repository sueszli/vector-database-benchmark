from google.cloud import vision_v1

def sample_get_product_set():
    if False:
        for i in range(10):
            print('nop')
    client = vision_v1.ProductSearchClient()
    request = vision_v1.GetProductSetRequest(name='name_value')
    response = client.get_product_set(request=request)
    print(response)