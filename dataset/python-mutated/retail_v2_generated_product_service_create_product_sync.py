from google.cloud import retail_v2

def sample_create_product():
    if False:
        i = 10
        return i + 15
    client = retail_v2.ProductServiceClient()
    product = retail_v2.Product()
    product.title = 'title_value'
    request = retail_v2.CreateProductRequest(parent='parent_value', product=product, product_id='product_id_value')
    response = client.create_product(request=request)
    print(response)