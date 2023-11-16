from google.cloud import compute_v1

def sample_get_from_family():
    if False:
        print('Hello World!')
    client = compute_v1.ImagesClient()
    request = compute_v1.GetFromFamilyImageRequest(family='family_value', project='project_value')
    response = client.get_from_family(request=request)
    print(response)