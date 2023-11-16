from google.cloud import vision_v1

def sample_delete_product():
    if False:
        print('Hello World!')
    client = vision_v1.ProductSearchClient()
    request = vision_v1.DeleteProductRequest(name='name_value')
    client.delete_product(request=request)