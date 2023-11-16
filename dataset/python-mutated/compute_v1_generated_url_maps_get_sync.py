from google.cloud import compute_v1

def sample_get():
    if False:
        return 10
    client = compute_v1.UrlMapsClient()
    request = compute_v1.GetUrlMapRequest(project='project_value', url_map='url_map_value')
    response = client.get(request=request)
    print(response)