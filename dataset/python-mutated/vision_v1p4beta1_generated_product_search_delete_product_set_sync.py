from google.cloud import vision_v1p4beta1

def sample_delete_product_set():
    if False:
        while True:
            i = 10
    client = vision_v1p4beta1.ProductSearchClient()
    request = vision_v1p4beta1.DeleteProductSetRequest(name='name_value')
    client.delete_product_set(request=request)