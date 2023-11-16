from google.cloud import vision_v1p4beta1

def sample_delete_reference_image():
    if False:
        return 10
    client = vision_v1p4beta1.ProductSearchClient()
    request = vision_v1p4beta1.DeleteReferenceImageRequest(name='name_value')
    client.delete_reference_image(request=request)