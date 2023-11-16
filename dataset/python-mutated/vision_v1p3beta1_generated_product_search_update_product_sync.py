from google.cloud import vision_v1p3beta1

def sample_update_product():
    if False:
        while True:
            i = 10
    client = vision_v1p3beta1.ProductSearchClient()
    request = vision_v1p3beta1.UpdateProductRequest()
    response = client.update_product(request=request)
    print(response)