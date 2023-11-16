from google.cloud import compute_v1

def sample_patch():
    if False:
        return 10
    client = compute_v1.ImagesClient()
    request = compute_v1.PatchImageRequest(image='image_value', project='project_value')
    response = client.patch(request=request)
    print(response)