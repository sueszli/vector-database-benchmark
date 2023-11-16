from google.cloud import vision_v1

def sample_delete_reference_image():
    if False:
        i = 10
        return i + 15
    client = vision_v1.ProductSearchClient()
    request = vision_v1.DeleteReferenceImageRequest(name='name_value')
    client.delete_reference_image(request=request)