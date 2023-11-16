from google.cloud import vision_v1p3beta1

def sample_list_product_sets():
    if False:
        while True:
            i = 10
    client = vision_v1p3beta1.ProductSearchClient()
    request = vision_v1p3beta1.ListProductSetsRequest(parent='parent_value')
    page_result = client.list_product_sets(request=request)
    for response in page_result:
        print(response)