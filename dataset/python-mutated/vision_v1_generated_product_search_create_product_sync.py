from google.cloud import vision_v1

def sample_create_product():
    if False:
        for i in range(10):
            print('nop')
    client = vision_v1.ProductSearchClient()
    request = vision_v1.CreateProductRequest(parent='parent_value')
    response = client.create_product(request=request)
    print(response)