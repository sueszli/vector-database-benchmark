from google.cloud import vision_v1p4beta1

def sample_list_reference_images():
    if False:
        print('Hello World!')
    client = vision_v1p4beta1.ProductSearchClient()
    request = vision_v1p4beta1.ListReferenceImagesRequest(parent='parent_value')
    page_result = client.list_reference_images(request=request)
    for response in page_result:
        print(response)