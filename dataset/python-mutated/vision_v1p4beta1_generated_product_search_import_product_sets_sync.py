from google.cloud import vision_v1p4beta1

def sample_import_product_sets():
    if False:
        i = 10
        return i + 15
    client = vision_v1p4beta1.ProductSearchClient()
    request = vision_v1p4beta1.ImportProductSetsRequest(parent='parent_value')
    operation = client.import_product_sets(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)