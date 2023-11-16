from google.cloud import retail_v2alpha

def sample_create_product():
    if False:
        while True:
            i = 10
    client = retail_v2alpha.ProductServiceClient()
    product = retail_v2alpha.Product()
    product.title = 'title_value'
    request = retail_v2alpha.CreateProductRequest(parent='parent_value', product=product, product_id='product_id_value')
    response = client.create_product(request=request)
    print(response)