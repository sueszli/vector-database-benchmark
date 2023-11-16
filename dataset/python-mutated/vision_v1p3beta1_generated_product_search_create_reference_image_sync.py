from google.cloud import vision_v1p3beta1

def sample_create_reference_image():
    if False:
        i = 10
        return i + 15
    client = vision_v1p3beta1.ProductSearchClient()
    reference_image = vision_v1p3beta1.ReferenceImage()
    reference_image.uri = 'uri_value'
    request = vision_v1p3beta1.CreateReferenceImageRequest(parent='parent_value', reference_image=reference_image)
    response = client.create_reference_image(request=request)
    print(response)