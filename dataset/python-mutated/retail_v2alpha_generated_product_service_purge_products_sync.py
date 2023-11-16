from google.cloud import retail_v2alpha

def sample_purge_products():
    if False:
        i = 10
        return i + 15
    client = retail_v2alpha.ProductServiceClient()
    request = retail_v2alpha.PurgeProductsRequest(parent='parent_value', filter='filter_value')
    operation = client.purge_products(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)