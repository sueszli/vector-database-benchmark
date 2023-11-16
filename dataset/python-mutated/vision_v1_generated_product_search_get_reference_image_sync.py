from google.cloud import vision_v1

def sample_get_reference_image():
    if False:
        return 10
    client = vision_v1.ProductSearchClient()
    request = vision_v1.GetReferenceImageRequest(name='name_value')
    response = client.get_reference_image(request=request)
    print(response)