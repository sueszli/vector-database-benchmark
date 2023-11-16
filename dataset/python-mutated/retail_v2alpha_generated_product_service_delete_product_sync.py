from google.cloud import retail_v2alpha

def sample_delete_product():
    if False:
        return 10
    client = retail_v2alpha.ProductServiceClient()
    request = retail_v2alpha.DeleteProductRequest(name='name_value')
    client.delete_product(request=request)