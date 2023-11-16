from google.cloud import retail_v2

def sample_update_product():
    if False:
        i = 10
        return i + 15
    client = retail_v2.ProductServiceClient()
    product = retail_v2.Product()
    product.title = 'title_value'
    request = retail_v2.UpdateProductRequest(product=product)
    response = client.update_product(request=request)
    print(response)