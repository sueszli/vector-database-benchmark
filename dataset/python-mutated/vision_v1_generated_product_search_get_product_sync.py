from google.cloud import vision_v1

def sample_get_product():
    if False:
        while True:
            i = 10
    client = vision_v1.ProductSearchClient()
    request = vision_v1.GetProductRequest(name='name_value')
    response = client.get_product(request=request)
    print(response)