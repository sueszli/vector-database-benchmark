from google.cloud import vision_v1p4beta1

def sample_create_product():
    if False:
        while True:
            i = 10
    client = vision_v1p4beta1.ProductSearchClient()
    request = vision_v1p4beta1.CreateProductRequest(parent='parent_value')
    response = client.create_product(request=request)
    print(response)