from google.cloud import retail_v2

def sample_delete_product():
    if False:
        i = 10
        return i + 15
    client = retail_v2.ProductServiceClient()
    request = retail_v2.DeleteProductRequest(name='name_value')
    client.delete_product(request=request)