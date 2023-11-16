from google.cloud import vision_v1

def sample_delete_product_set():
    if False:
        for i in range(10):
            print('nop')
    client = vision_v1.ProductSearchClient()
    request = vision_v1.DeleteProductSetRequest(name='name_value')
    client.delete_product_set(request=request)