from google.cloud import vision_v1p4beta1

def sample_update_product():
    if False:
        for i in range(10):
            print('nop')
    client = vision_v1p4beta1.ProductSearchClient()
    request = vision_v1p4beta1.UpdateProductRequest()
    response = client.update_product(request=request)
    print(response)