from google.cloud import vision_v1p4beta1

def sample_purge_products():
    if False:
        while True:
            i = 10
    client = vision_v1p4beta1.ProductSearchClient()
    request = vision_v1p4beta1.PurgeProductsRequest(parent='parent_value')
    operation = client.purge_products(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)