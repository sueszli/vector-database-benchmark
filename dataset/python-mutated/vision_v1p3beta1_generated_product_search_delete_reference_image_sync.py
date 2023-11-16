from google.cloud import vision_v1p3beta1

def sample_delete_reference_image():
    if False:
        for i in range(10):
            print('nop')
    client = vision_v1p3beta1.ProductSearchClient()
    request = vision_v1p3beta1.DeleteReferenceImageRequest(name='name_value')
    client.delete_reference_image(request=request)