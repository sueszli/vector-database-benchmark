from google.cloud import vision_v1

def sample_create_reference_image():
    if False:
        print('Hello World!')
    client = vision_v1.ProductSearchClient()
    reference_image = vision_v1.ReferenceImage()
    reference_image.uri = 'uri_value'
    request = vision_v1.CreateReferenceImageRequest(parent='parent_value', reference_image=reference_image)
    response = client.create_reference_image(request=request)
    print(response)