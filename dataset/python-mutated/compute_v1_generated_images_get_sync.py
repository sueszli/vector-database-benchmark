from google.cloud import compute_v1

def sample_get():
    if False:
        while True:
            i = 10
    client = compute_v1.ImagesClient()
    request = compute_v1.GetImageRequest(image='image_value', project='project_value')
    response = client.get(request=request)
    print(response)