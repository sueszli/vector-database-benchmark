from google.cloud import retail_v2alpha

def sample_update_product():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2alpha.ProductServiceClient()
    product = retail_v2alpha.Product()
    product.title = 'title_value'
    request = retail_v2alpha.UpdateProductRequest(product=product)
    response = client.update_product(request=request)
    print(response)