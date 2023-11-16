from google.cloud import retail_v2beta

def sample_import_products():
    if False:
        print('Hello World!')
    client = retail_v2beta.ProductServiceClient()
    input_config = retail_v2beta.ProductInputConfig()
    input_config.product_inline_source.products.title = 'title_value'
    request = retail_v2beta.ImportProductsRequest(parent='parent_value', input_config=input_config)
    operation = client.import_products(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)