from google.cloud import vision_v1p3beta1

def sample_import_product_sets():
    if False:
        while True:
            i = 10
    client = vision_v1p3beta1.ProductSearchClient()
    request = vision_v1p3beta1.ImportProductSetsRequest(parent='parent_value')
    operation = client.import_product_sets(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)