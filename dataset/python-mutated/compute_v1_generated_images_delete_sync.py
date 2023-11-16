from google.cloud import compute_v1

def sample_delete():
    if False:
        i = 10
        return i + 15
    client = compute_v1.ImagesClient()
    request = compute_v1.DeleteImageRequest(image='image_value', project='project_value')
    response = client.delete(request=request)
    print(response)