from google.cloud import vision_v1p4beta1

def sample_update_product_set():
    if False:
        print('Hello World!')
    client = vision_v1p4beta1.ProductSearchClient()
    request = vision_v1p4beta1.UpdateProductSetRequest()
    response = client.update_product_set(request=request)
    print(response)