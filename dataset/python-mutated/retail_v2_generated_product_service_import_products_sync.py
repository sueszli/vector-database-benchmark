from google.cloud import retail_v2

def sample_import_products():
    if False:
        while True:
            i = 10
    client = retail_v2.ProductServiceClient()
    input_config = retail_v2.ProductInputConfig()
    input_config.product_inline_source.products.title = 'title_value'
    request = retail_v2.ImportProductsRequest(parent='parent_value', input_config=input_config)
    operation = client.import_products(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)