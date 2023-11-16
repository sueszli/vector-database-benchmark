from google.cloud import retail_v2beta

def sample_create_product():
    if False:
        print('Hello World!')
    client = retail_v2beta.ProductServiceClient()
    product = retail_v2beta.Product()
    product.title = 'title_value'
    request = retail_v2beta.CreateProductRequest(parent='parent_value', product=product, product_id='product_id_value')
    response = client.create_product(request=request)
    print(response)