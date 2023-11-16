from google.cloud import vision_v1p4beta1

def sample_create_product_set():
    if False:
        return 10
    client = vision_v1p4beta1.ProductSearchClient()
    request = vision_v1p4beta1.CreateProductSetRequest(parent='parent_value')
    response = client.create_product_set(request=request)
    print(response)