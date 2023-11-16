from google.cloud import compute_v1

def sample_deprecate():
    if False:
        return 10
    client = compute_v1.ImagesClient()
    request = compute_v1.DeprecateImageRequest(image='image_value', project='project_value')
    response = client.deprecate(request=request)
    print(response)