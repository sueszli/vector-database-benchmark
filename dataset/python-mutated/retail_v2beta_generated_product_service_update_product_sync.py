from google.cloud import retail_v2beta

def sample_update_product():
    if False:
        while True:
            i = 10
    client = retail_v2beta.ProductServiceClient()
    product = retail_v2beta.Product()
    product.title = 'title_value'
    request = retail_v2beta.UpdateProductRequest(product=product)
    response = client.update_product(request=request)
    print(response)