from google.cloud import compute_v1

def sample_patch():
    if False:
        return 10
    client = compute_v1.UrlMapsClient()
    request = compute_v1.PatchUrlMapRequest(project='project_value', url_map='url_map_value')
    response = client.patch(request=request)
    print(response)