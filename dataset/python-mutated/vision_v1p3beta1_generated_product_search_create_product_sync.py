from google.cloud import vision_v1p3beta1

def sample_create_product():
    if False:
        print('Hello World!')
    client = vision_v1p3beta1.ProductSearchClient()
    request = vision_v1p3beta1.CreateProductRequest(parent='parent_value')
    response = client.create_product(request=request)
    print(response)