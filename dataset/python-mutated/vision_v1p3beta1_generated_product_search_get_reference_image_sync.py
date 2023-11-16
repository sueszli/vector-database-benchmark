from google.cloud import vision_v1p3beta1

def sample_get_reference_image():
    if False:
        print('Hello World!')
    client = vision_v1p3beta1.ProductSearchClient()
    request = vision_v1p3beta1.GetReferenceImageRequest(name='name_value')
    response = client.get_reference_image(request=request)
    print(response)