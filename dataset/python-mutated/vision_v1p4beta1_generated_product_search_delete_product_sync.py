from google.cloud import vision_v1p4beta1

def sample_delete_product():
    if False:
        return 10
    client = vision_v1p4beta1.ProductSearchClient()
    request = vision_v1p4beta1.DeleteProductRequest(name='name_value')
    client.delete_product(request=request)